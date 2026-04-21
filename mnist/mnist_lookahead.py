import copy
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import pandas as pd

from mnist_plotting_utils import (
    ensure_dir,
    query_records_to_dataframe,
    save_dataframe,
    plot_accuracy_curves,
    plot_uncertainty_curves,
    plot_runtime_curves,
    plot_query_trajectory,
)


# ============================================================
# Small CNN used by every member of the ensemble.
# ============================================================

class SmallMNISTNet(nn.Module):
    """
    A deliberately small classifier so experiments run quickly.

    Two design choices matter here:
    1. We return *features* from a penultimate layer. Those features are used
       for the query-diversity distance inside the multi-sampling policy.
    2. We return *logits* for classification and for pseudo-label generation.
    """

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, feature_dim)
        self.fc2 = nn.Linear(feature_dim, 10)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.fc2(features)
        return logits


# ============================================================
# Data state objects.
# ============================================================

@dataclass
class RealState:
    """
    The *real* active-learning state that persists across outer acquisition steps.

    labeled_true_indices:
        The indices whose true MNIST labels we are allowed to use.

    unlabeled_pool_indices:
        The candidate grid / pool the policy may choose from.
    """

    labeled_true_indices: List[int]
    unlabeled_pool_indices: List[int]


@dataclass
class SimulatedState:
    """
    State used inside the *lookahead tree*.

    This mirrors your original idea:
    - inside the tree, expansions are pseudo-labeled using the current ensemble
    - only the final chosen root query gets committed with a true label outside
      the tree

    pseudo_logits_by_index:
        Maps dataset index -> ensemble-averaged logits generated during search.
    """

    labeled_true_indices: List[int]
    unlabeled_pool_indices: List[int]
    pseudo_logits_by_index: Dict[int, torch.Tensor] = field(default_factory=dict)


# ============================================================
# Mixed supervision dataset.
# ============================================================

class MixedMNISTDataset(Dataset):
    """
    Dataset that mixes:
    - truly labeled MNIST examples
    - pseudo-labeled examples represented by target logits

    For true labels we use standard cross-entropy.
    For pseudo labels we use MSE to match target logits.

    This is a direct simplification of your fuel-sequence idea:
    instead of pseudo-labeling a whole trajectory, we pseudo-label one MNIST image
    with ensemble-average logits.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        true_indices: Sequence[int],
        pseudo_logits_by_index: Dict[int, torch.Tensor],
    ):
        self.base_dataset = base_dataset
        self.true_indices = list(true_indices)
        self.pseudo_indices = list(pseudo_logits_by_index.keys())
        self.pseudo_logits_by_index = {
            int(k): v.detach().cpu().clone().float() for k, v in pseudo_logits_by_index.items()
        }
        self.all_indices = self.true_indices + self.pseudo_indices

    def __len__(self) -> int:
        return len(self.all_indices)

    def __getitem__(self, idx: int):
        base_idx = self.all_indices[idx]
        image, true_label = self.base_dataset[base_idx]

        # use_pseudo = 0 for true-label examples, 1 for pseudo-labeled examples
        if base_idx in self.pseudo_logits_by_index:
            target_logits = self.pseudo_logits_by_index[base_idx]
            use_pseudo = 1
            true_label = int(true_label)
        else:
            target_logits = torch.zeros(10, dtype=torch.float32)
            use_pseudo = 0
            true_label = int(true_label)

        return image, torch.tensor(true_label, dtype=torch.long), target_logits, torch.tensor(use_pseudo)


# ============================================================
# Ensemble wrapper.
# ============================================================

class ClassifierEnsemble(nn.Module):
    """
    Thin wrapper around multiple SmallMNISTNet models.

    Key methods:
    - member_logits: all member predictions
    - average_logits: mean logits across members
    - uncertainty_scores: per-sample disagreement score
    - average_features: mean embedding for distance computations
    """

    def __init__(self, members: List[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    @property
    def num_members(self) -> int:
        return len(self.members)

    def member_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = [m(x).unsqueeze(0) for m in self.members]
        return torch.cat(logits, dim=0)  # [M, B, 10]

    def average_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.member_logits(x).mean(dim=0)  # [B, 10]

    def uncertainty_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple ensemble disagreement score.

        We use mean variance across the 10 logits.
        This mirrors your original use of ensemble variance as the node score.
        """
        logits = self.member_logits(x)
        return logits.var(dim=0).mean(dim=1)  # [B]

    def average_features(self, x: torch.Tensor) -> torch.Tensor:
        features = [m.extract_features(x).unsqueeze(0) for m in self.members]
        return torch.cat(features, dim=0).mean(dim=0)  # [B, D]


# ============================================================
# Training helpers.
# ============================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_ensemble_accuracy(ensemble: ClassifierEnsemble, loader: DataLoader, device: torch.device) -> float:
    ensemble.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = ensemble.average_logits(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)



def train_single_model(
    model: nn.Module,
    dataset: MixedMNISTDataset,
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    pseudo_loss_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Train one classifier on a mix of true labels and pseudo logits.

    Loss:
    - true-label examples: cross entropy on integer labels
    - pseudo-labeled examples: MSE on logits against ensemble-average target logits

    Returning metrics here makes the outer search code easy to instrument.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_true_ce = 0.0
    total_pseudo_mse = 0.0
    total_batches = 0

    model.to(device)
    model.train()

    for _ in range(epochs):
        for images, true_labels, target_logits, use_pseudo in loader:
            images = images.to(device)
            true_labels = true_labels.to(device)
            target_logits = target_logits.to(device)
            use_pseudo = use_pseudo.to(device)

            logits = model(images)
            loss = torch.tensor(0.0, device=device)

            # True-label loss.
            true_mask = use_pseudo == 0
            if true_mask.any():
                ce_loss = F.cross_entropy(logits[true_mask], true_labels[true_mask])
                loss = loss + ce_loss
                total_true_ce += float(ce_loss.item())

            # Pseudo-label loss.
            pseudo_mask = use_pseudo == 1
            if pseudo_mask.any():
                mse_loss = F.mse_loss(logits[pseudo_mask], target_logits[pseudo_mask])
                loss = loss + pseudo_loss_weight * mse_loss
                total_pseudo_mse += float(mse_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_batches += 1

    return {
        "mean_true_ce": total_true_ce / max(total_batches, 1),
        "mean_pseudo_mse": total_pseudo_mse / max(total_batches, 1),
    }



def train_ensemble(
    train_dataset: MixedMNISTDataset,
    test_loader: DataLoader,
    device: torch.device,
    num_members: int = 3,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    pseudo_loss_weight: float = 1.0,
    seed: int = 0,
) -> Tuple[ClassifierEnsemble, Dict[str, float]]:
    """
    Train an ensemble using bootstrap resampling over the current mixed dataset.

    Bootstrap resampling helps the members disagree a bit more, which is useful
    because the whole search procedure relies on disagreement as an uncertainty signal.
    """
    members = []
    member_metrics = []
    n = len(train_dataset)

    for member_id in range(num_members):
        set_seed(seed + 1000 * member_id)

        # Simple bootstrap sample.
        bootstrap_indices = np.random.choice(n, size=n, replace=True).tolist()
        bootstrap_dataset = torch.utils.data.Subset(train_dataset, bootstrap_indices)

        model = SmallMNISTNet()
        metrics = train_single_model(
            model=model,
            dataset=bootstrap_dataset,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            pseudo_loss_weight=pseudo_loss_weight,
        )
        members.append(model)
        member_metrics.append(metrics)

    ensemble = ClassifierEnsemble(members).to(device)
    test_acc = evaluate_ensemble_accuracy(ensemble, test_loader, device)

    summary = {
        "test_accuracy": test_acc,
        "mean_true_ce": float(np.mean([m["mean_true_ce"] for m in member_metrics])),
        "mean_pseudo_mse": float(np.mean([m["mean_pseudo_mse"] for m in member_metrics])),
    }
    return ensemble, summary


# ============================================================
# Node / search helpers.
# ============================================================

class Node:
    """
    Tree node for simulated lookahead.

    Each node represents:
    - a simulated dataset state
    - an ensemble trained on that simulated state
    - a scalar score equal to mean pool uncertainty under that ensemble
    """

    _next_id = 0

    def __init__(
        self,
        score: float,
        sim_state: SimulatedState,
        ensemble: ClassifierEnsemble,
        grid_scores: Optional[Dict[int, float]] = None,
        query_index: Optional[int] = None,
        depth: int = 0,
        used_indices: Optional[List[int]] = None,
    ):
        self.node_id = Node._next_id
        Node._next_id += 1
        self.score = float(score)
        self.sim_state = sim_state
        self.ensemble = ensemble
        self.grid_scores = {} if grid_scores is None else dict(grid_scores)
        self.query_index = query_index
        self.depth = depth
        self.used_indices = [] if used_indices is None else list(used_indices)
        self.children: List[Node] = []

    def add_child(self, child: "Node"):
        self.children.append(child)


@torch.no_grad()
def compute_grid_scores(
    ensemble: ClassifierEnsemble,
    base_dataset: Dataset,
    candidate_indices: Sequence[int],
    device: torch.device,
    batch_size: int = 256,
) -> Dict[int, float]:
    """
    Score every pool element with the ensemble disagreement score.

    This is the MNIST version of your original grid scan.
    Here the grid is literally the unlabeled pool images.
    """
    ensemble.eval()
    scores = {}

    loader = DataLoader(Subset(base_dataset, list(candidate_indices)), batch_size=batch_size, shuffle=False)
    ordered_indices = list(candidate_indices)
    cursor = 0

    for images, _ in loader:
        images = images.to(device)
        batch_scores = ensemble.uncertainty_scores(images).detach().cpu().tolist()
        for score in batch_scores:
            scores[ordered_indices[cursor]] = float(score)
            cursor += 1

    return scores


@torch.no_grad()
def compute_feature_embeddings(
    ensemble: ClassifierEnsemble,
    base_dataset: Dataset,
    candidate_indices: Sequence[int],
    device: torch.device,
    batch_size: int = 256,
) -> Dict[int, torch.Tensor]:
    """
    Build one feature embedding per candidate image.

    We use mean penultimate-layer features across ensemble members.
    Those features drive the diversity metric inside the policy.
    """
    ensemble.eval()
    feats = {}

    loader = DataLoader(Subset(base_dataset, list(candidate_indices)), batch_size=batch_size, shuffle=False)
    ordered_indices = list(candidate_indices)
    cursor = 0

    for images, _ in loader:
        images = images.to(device)
        batch_feats = ensemble.average_features(images).detach().cpu()
        for row in batch_feats:
            feats[ordered_indices[cursor]] = row.clone()
            cursor += 1

    return feats


def feature_distance(feature_a: torch.Tensor, feature_b: torch.Tensor) -> float:
    """
    Distance used for diversity inside the multi-sampling policy.

    We use cosine distance in feature space:
        1 - cosine_similarity

    That is a much better MNIST-specific proxy than the original simple metric,
    because it measures semantic similarity according to the classifier features.
    """
    a = feature_a.float()
    b = feature_b.float()
    denom = (a.norm() * b.norm()).clamp_min(1e-8)
    cos_sim = float(torch.dot(a, b) / denom)
    return 1.0 - cos_sim



def r_hat_from_grid_scores(grid_scores: Dict[int, float]) -> float:
    if len(grid_scores) == 0:
        return 0.0
    return float(np.mean(list(grid_scores.values())))



def policy_select_indices(
    grid_scores: Dict[int, float],
    feature_cache: Dict[int, torch.Tensor],
    n_queries: int = 3,
    lmbda: float = 1.0,
    forbidden_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Greedy top-k query proposal policy.

    Base term:
        ensemble disagreement on the candidate image

    Diversity term:
        lambda * feature-space distance from already selected candidates

    This mirrors the structure of your original policy, but now the distance is
    based on classifier features instead of the old cheap metric.
    """
    forbidden = set([] if forbidden_indices is None else list(forbidden_indices))
    chosen: List[int] = []
    candidate_indices = list(grid_scores.keys())

    for _ in range(min(n_queries, len(candidate_indices))):
        best_idx = None
        best_value = -float("inf")

        for idx in candidate_indices:
            if idx in forbidden or idx in chosen:
                continue

            value = float(grid_scores[idx])
            for prev_idx in chosen:
                value += lmbda * feature_distance(feature_cache[idx], feature_cache[prev_idx])

            if value > best_value:
                best_value = value
                best_idx = idx

        if best_idx is None:
            break
        chosen.append(best_idx)

    return chosen


@torch.no_grad()
def pseudo_label_index(
    ensemble: ClassifierEnsemble,
    base_dataset: Dataset,
    data_index: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Return the ensemble-average logits for one image.

    This is the simplified MNIST pseudo-label corresponding to your original
    trajectory pseudo-label idea.
    """
    image, _ = base_dataset[data_index]
    image = image.unsqueeze(0).to(device)
    avg_logits = ensemble.average_logits(image).squeeze(0).detach().cpu()
    return avg_logits



def simulate_environment_step(
    node: Node,
    chosen_index: int,
    base_train_dataset: Dataset,
    test_loader: DataLoader,
    device: torch.device,
    num_members: int,
    train_epochs: int,
    batch_size: int,
    lr: float,
    pseudo_loss_weight: float,
    seed: int,
) -> Tuple[SimulatedState, ClassifierEnsemble, Dict[int, float], Dict[int, torch.Tensor], Dict[str, float]]:
    """
    One simulated lookahead step.

    Inside the tree:
    - pseudo-label the chosen image using average ensemble logits
    - add that image to the simulated dataset with pseudo logits
    - retrain the ensemble on the simulated dataset
    - rescore the remaining pool
    """
    pseudo_logits = pseudo_label_index(node.ensemble, base_train_dataset, chosen_index, device)

    new_pseudo_logits = dict(node.sim_state.pseudo_logits_by_index)
    new_pseudo_logits[chosen_index] = pseudo_logits

    new_pool = [idx for idx in node.sim_state.unlabeled_pool_indices if idx != chosen_index]

    new_state = SimulatedState(
        labeled_true_indices=list(node.sim_state.labeled_true_indices),
        unlabeled_pool_indices=new_pool,
        pseudo_logits_by_index=new_pseudo_logits,
    )

    mixed_dataset = MixedMNISTDataset(
        base_dataset=base_train_dataset,
        true_indices=new_state.labeled_true_indices,
        pseudo_logits_by_index=new_state.pseudo_logits_by_index,
    )

    ensemble, train_metrics = train_ensemble(
        train_dataset=mixed_dataset,
        test_loader=test_loader,
        device=device,
        num_members=num_members,
        epochs=train_epochs,
        batch_size=batch_size,
        lr=lr,
        pseudo_loss_weight=pseudo_loss_weight,
        seed=seed,
    )

    grid_scores = compute_grid_scores(ensemble, base_train_dataset, new_pool, device=device, batch_size=batch_size)
    feature_cache = compute_feature_embeddings(ensemble, base_train_dataset, new_pool, device=device, batch_size=batch_size)

    return new_state, ensemble, grid_scores, feature_cache, train_metrics



def subtree_value(node: Node, mode: str = "terminal") -> float:
    if not node.children:
        return float(node.score)
    child_vals = [subtree_value(child, mode=mode) for child in node.children]
    if mode == "terminal":
        return max(float(node.score), max(child_vals))
    if mode == "cumulative":
        return float(node.score) + max(child_vals)
    raise ValueError(f"Unknown mode: {mode}")



def best_root_child(root: Node, mode: str = "terminal") -> Optional[Node]:
    if not root.children:
        return None
    return max(root.children, key=lambda child: subtree_value(child, mode=mode))



def next_state_select(root: Node, mode: str = "terminal") -> Node:
    if not root.children:
        return root
    node = best_root_child(root, mode=mode)
    while node.children:
        node = max(node.children, key=lambda child: subtree_value(child, mode=mode))
    return node


# ============================================================
# Full lookahead acquisition.
# ============================================================


def choose_query_with_lookahead(
    real_state: RealState,
    base_train_dataset: Dataset,
    test_loader: DataLoader,
    device: torch.device,
    num_search_iters: int = 3,
    n_policy_queries: int = 3,
    lmbda: float = 0.5,
    backup_mode: str = "terminal",
    num_members: int = 3,
    train_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    pseudo_loss_weight: float = 1.0,
    seed: int = 0,
) -> Tuple[int, Dict[str, object]]:
    """
    Build a lookahead tree, then return the best *root child*.

    Crucially:
    - tree expansions use pseudo-labels
    - the final returned root child is only a *recommendation*
    - the caller decides whether to commit that image with its true label
    """
    sim_state = SimulatedState(
        labeled_true_indices=list(real_state.labeled_true_indices),
        unlabeled_pool_indices=list(real_state.unlabeled_pool_indices),
        pseudo_logits_by_index={},
    )

    # Root ensemble is trained only on the currently true-labeled data.
    root_train_dataset = MixedMNISTDataset(
        base_dataset=base_train_dataset,
        true_indices=sim_state.labeled_true_indices,
        pseudo_logits_by_index={},
    )
    root_ensemble, root_train_metrics = train_ensemble(
        train_dataset=root_train_dataset,
        test_loader=test_loader,
        device=device,
        num_members=num_members,
        epochs=train_epochs,
        batch_size=batch_size,
        lr=lr,
        pseudo_loss_weight=pseudo_loss_weight,
        seed=seed,
    )

    root_grid_scores = compute_grid_scores(
        root_ensemble,
        base_train_dataset,
        sim_state.unlabeled_pool_indices,
        device=device,
        batch_size=batch_size,
    )
    root_feature_cache = compute_feature_embeddings(
        root_ensemble,
        base_train_dataset,
        sim_state.unlabeled_pool_indices,
        device=device,
        batch_size=batch_size,
    )
    root_score = r_hat_from_grid_scores(root_grid_scores)
    root = Node(
        score=root_score,
        sim_state=sim_state,
        ensemble=root_ensemble,
        grid_scores=root_grid_scores,
        query_index=None,
        depth=0,
        used_indices=[],
    )

    history = []
    current_node = root

    for search_iter in range(num_search_iters):
        iter_start = time.perf_counter()

        candidate_indices = policy_select_indices(
            grid_scores=current_node.grid_scores,
            feature_cache=root_feature_cache if current_node.depth == 0 else compute_feature_embeddings(
                current_node.ensemble,
                base_train_dataset,
                current_node.sim_state.unlabeled_pool_indices,
                device=device,
                batch_size=batch_size,
            ),
            n_queries=n_policy_queries,
            lmbda=lmbda,
            forbidden_indices=current_node.used_indices,
        )

        expansion_summaries = []

        if not current_node.children:
            for local_rank, chosen_index in enumerate(candidate_indices, start=1):
                child_state, child_ensemble, child_grid_scores, child_feature_cache, child_train_metrics = simulate_environment_step(
                    node=current_node,
                    chosen_index=chosen_index,
                    base_train_dataset=base_train_dataset,
                    test_loader=test_loader,
                    device=device,
                    num_members=num_members,
                    train_epochs=train_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    pseudo_loss_weight=pseudo_loss_weight,
                    seed=seed + 10 * search_iter + local_rank,
                )

                child = Node(
                    score=r_hat_from_grid_scores(child_grid_scores),
                    sim_state=child_state,
                    ensemble=child_ensemble,
                    grid_scores=child_grid_scores,
                    query_index=chosen_index,
                    depth=current_node.depth + 1,
                    used_indices=current_node.used_indices + [chosen_index],
                )
                current_node.add_child(child)

                expansion_summaries.append({
                    "search_iter": search_iter,
                    "rank_in_policy_list": local_rank,
                    "query_index": chosen_index,
                    "child_score": float(child.score),
                    "child_test_accuracy": float(child_train_metrics["test_accuracy"]),
                    "child_mean_true_ce": float(child_train_metrics["mean_true_ce"]),
                    "child_mean_pseudo_mse": float(child_train_metrics["mean_pseudo_mse"]),
                })

        current_node = next_state_select(root, mode=backup_mode)
        best_root = best_root_child(root, mode=backup_mode)
        iter_time = time.perf_counter() - iter_start

        history.append({
            "search_iter": search_iter,
            "selected_node_depth_after": current_node.depth,
            "best_root_query_index": None if best_root is None else int(best_root.query_index),
            "best_root_subtree_value": np.nan if best_root is None else float(subtree_value(best_root, mode=backup_mode)),
            "best_leaf_score_after": float(current_node.score),
            "search_iter_seconds": float(iter_time),
            "num_candidates_considered": len(candidate_indices),
            "expansions": expansion_summaries,
        })

        if best_root is not None:
            root_feature_cache = compute_feature_embeddings(
                root_ensemble,
                base_train_dataset,
                root.sim_state.unlabeled_pool_indices,
                device=device,
                batch_size=batch_size,
            )

    chosen_root = best_root_child(root, mode=backup_mode)
    if chosen_root is None:
        fallback = random.choice(real_state.unlabeled_pool_indices)
        return fallback, {"search_history": history, "root_score": root_score}

    return int(chosen_root.query_index), {
        "search_history": history,
        "root_score": root_score,
        "chosen_root_subtree_value": float(subtree_value(chosen_root, mode=backup_mode)),
    }


# ============================================================
# Baselines.
# ============================================================


def choose_query_random(real_state: RealState, rng: random.Random) -> Tuple[int, Dict[str, object]]:
    idx = rng.choice(real_state.unlabeled_pool_indices)
    return idx, {"strategy": "random"}



def choose_query_uncertainty(
    real_state: RealState,
    base_train_dataset: Dataset,
    test_loader: DataLoader,
    device: torch.device,
    num_members: int = 3,
    train_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
) -> Tuple[int, Dict[str, object]]:
    """
    Baseline: train on currently true-labeled data, then pick the pool image with
    maximum ensemble disagreement.
    """
    dataset = MixedMNISTDataset(base_train_dataset, real_state.labeled_true_indices, {})
    ensemble, train_metrics = train_ensemble(
        train_dataset=dataset,
        test_loader=test_loader,
        device=device,
        num_members=num_members,
        epochs=train_epochs,
        batch_size=batch_size,
        lr=lr,
        pseudo_loss_weight=1.0,
        seed=seed,
    )
    grid_scores = compute_grid_scores(ensemble, base_train_dataset, real_state.unlabeled_pool_indices, device=device, batch_size=batch_size)
    best_idx = max(grid_scores.keys(), key=lambda idx: grid_scores[idx])
    return int(best_idx), {
        "strategy": "uncertainty",
        "root_score": r_hat_from_grid_scores(grid_scores),
        "train_metrics": train_metrics,
    }


# ============================================================
# Outer active-learning loop.
# ============================================================


def build_dataloaders(data_root: str = "./data", batch_size: int = 256):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, test_loader



def initialize_real_state(
    train_dataset: Dataset,
    initial_labeled_size: int,
    pool_size: int,
    seed: int,
) -> RealState:
    """
    Build D_0 from a random subset of MNIST.

    We keep the pool smaller than the full 60k training set by default so the
    experiments run in reasonable time. If you want the literal full training set
    as the grid, set pool_size to len(train_dataset).
    """
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(train_dataset))
    rng.shuffle(all_indices)

    chosen_pool = all_indices[:pool_size].tolist()
    labeled = chosen_pool[:initial_labeled_size]
    unlabeled = chosen_pool[initial_labeled_size:]
    return RealState(labeled_true_indices=labeled, unlabeled_pool_indices=unlabeled)



def commit_true_label(state: RealState, chosen_index: int) -> RealState:
    new_labeled = list(state.labeled_true_indices) + [int(chosen_index)]
    new_pool = [idx for idx in state.unlabeled_pool_indices if idx != chosen_index]
    return RealState(labeled_true_indices=new_labeled, unlabeled_pool_indices=new_pool)



def fit_and_evaluate_real_state(
    state: RealState,
    base_train_dataset: Dataset,
    test_loader: DataLoader,
    device: torch.device,
    num_members: int,
    train_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Tuple[ClassifierEnsemble, Dict[int, float], Dict[str, float]]:
    """
    Train the *real* ensemble on currently true-labeled data only, then evaluate it.
    """
    dataset = MixedMNISTDataset(base_train_dataset, state.labeled_true_indices, {})
    ensemble, train_metrics = train_ensemble(
        train_dataset=dataset,
        test_loader=test_loader,
        device=device,
        num_members=num_members,
        epochs=train_epochs,
        batch_size=batch_size,
        lr=lr,
        pseudo_loss_weight=1.0,
        seed=seed,
    )

    pool_scores = compute_grid_scores(
        ensemble,
        base_train_dataset,
        state.unlabeled_pool_indices,
        device=device,
        batch_size=batch_size,
    )
    summary = dict(train_metrics)
    summary["pool_mean_uncertainty"] = r_hat_from_grid_scores(pool_scores)
    summary["pool_max_uncertainty"] = float(max(pool_scores.values())) if len(pool_scores) > 0 else 0.0
    return ensemble, pool_scores, summary



def run_strategy_experiment(
    strategy_name: str,
    train_dataset: Dataset,
    test_loader: DataLoader,
    device: torch.device,
    initial_labeled_size: int = 100,
    pool_size: int = 5000,
    acquisition_steps: int = 10,
    num_members: int = 3,
    train_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    lookahead_search_iters: int = 3,
    lookahead_branching: int = 3,
    lookahead_lambda: float = 0.5,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one full active-learning experiment.

    At each *real* acquisition step:
    1. Train/evaluate the current real model using only true labels.
    2. Choose one pool image using the requested strategy.
    3. Commit that image with its TRUE label.

    This is exactly what makes MNIST useful here:
    the tree search can use pseudo-labels internally, but the outer loop can still
    be evaluated against real labels and compared to baselines.
    """
    rng = random.Random(seed)
    state = initialize_real_state(train_dataset, initial_labeled_size, pool_size, seed)

    history_rows = []
    query_rows = []

    for step in range(acquisition_steps + 1):
        outer_start = time.perf_counter()
        print(f"\rAquisition Step : {step}",end="")

        # Evaluate the current true-labeled state *before* adding the next label.
        ensemble, pool_scores, eval_metrics = fit_and_evaluate_real_state(
            state=state,
            base_train_dataset=train_dataset,
            test_loader=test_loader,
            device=device,
            num_members=num_members,
            train_epochs=train_epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed + 17 * step,
        )

        row = {
            "strategy": strategy_name,
            "step": step,
            "num_true_labels": len(state.labeled_true_indices),
            "num_unlabeled_pool": len(state.unlabeled_pool_indices),
            "test_accuracy": float(eval_metrics["test_accuracy"]),
            "pool_mean_uncertainty": float(eval_metrics["pool_mean_uncertainty"]),
            "pool_max_uncertainty": float(eval_metrics["pool_max_uncertainty"]),
            "mean_true_ce": float(eval_metrics["mean_true_ce"]),
            "mean_pseudo_mse": float(eval_metrics["mean_pseudo_mse"]),
        }

        # Stop after recording the final post-acquisition metric point.
        if step == acquisition_steps or len(state.unlabeled_pool_indices) == 0:
            row["outer_step_seconds"] = float(time.perf_counter() - outer_start)
            history_rows.append(row)
            break

        # Choose the next real acquisition.
        if strategy_name == "random":
            chosen_index, strategy_info = choose_query_random(state, rng)
        elif strategy_name == "uncertainty":
            chosen_index, strategy_info = choose_query_uncertainty(
                real_state=state,
                base_train_dataset=train_dataset,
                test_loader=test_loader,
                device=device,
                num_members=num_members,
                train_epochs=train_epochs,
                batch_size=batch_size,
                lr=lr,
                seed=seed + 100 * step,
            )
        elif strategy_name == "lookahead":
            chosen_index, strategy_info = choose_query_with_lookahead(
                real_state=state,
                base_train_dataset=train_dataset,
                test_loader=test_loader,
                device=device,
                num_search_iters=lookahead_search_iters,
                n_policy_queries=lookahead_branching,
                lmbda=lookahead_lambda,
                backup_mode="terminal",
                num_members=num_members,
                train_epochs=train_epochs,
                batch_size=batch_size,
                lr=lr,
                pseudo_loss_weight=1.0,
                seed=seed + 200 * step,
            )
        else:
            raise ValueError(f"Unknown strategy_name={strategy_name}")

        # Record the chosen real query so we can visualize acquisition trajectories.
        img, true_label = train_dataset[chosen_index]
        query_rows.append({
            "strategy": strategy_name,
            "step": step,
            "chosen_index": int(chosen_index),
            "true_label": int(true_label),
            "uncertainty_at_choice": float(pool_scores.get(chosen_index, np.nan)),
            # Simple image-space summary for plotting. These are not the search
            # features; they are just lightweight coordinates for visualization.
            "pixel_mean": float(img.mean().item()),
            "pixel_std": float(img.std().item()),
        })

        row["chosen_index"] = int(chosen_index)
        row["chosen_label"] = int(true_label)
        row["outer_step_seconds"] = float(time.perf_counter() - outer_start)
        history_rows.append(row)

        # Commit the winner with its TRUE label.
        state = commit_true_label(state, chosen_index)
    print("\n aquisition concluded")
    return pd.DataFrame(history_rows), pd.DataFrame(query_rows)


# ============================================================
# Script entrypoint.
# ============================================================


def main():
    """
    Example experiment setup.

    The defaults below are intentionally modest so the code is practical to run:
    - pool_size smaller than full MNIST train set
    - small ensemble
    - few train epochs
    - few acquisition steps

    Once the pipeline works, you can scale up any of those.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    set_seed(seed)

    train_dataset, test_loader = build_dataloaders(batch_size=256)

    artifacts_dir = "mnist_lookahead_artifacts"
    ensure_dir(artifacts_dir)

    strategy_histories = []
    strategy_queries = []

    for strategy in ["random", "uncertainty", "lookahead"]:
        print(f"\nRunning strategy: {strategy}")
        history_df, query_df = run_strategy_experiment(
            strategy_name=strategy,
            train_dataset=train_dataset,
            test_loader=test_loader,
            device=device,
            initial_labeled_size=100,
            pool_size=5000,
            acquisition_steps=100,
            num_members=3,
            train_epochs=2,
            batch_size=64,
            lr=1e-3,
            lookahead_search_iters=2,
            lookahead_branching=3,
            lookahead_lambda=0.35,
            seed=seed,
        )
        strategy_histories.append(history_df)
        strategy_queries.append(query_df)

    all_history = pd.concat(strategy_histories, ignore_index=True)
    all_queries = pd.concat(strategy_queries, ignore_index=True)

    save_dataframe(all_history, os.path.join(artifacts_dir, "mnist_strategy_history.csv"))
    save_dataframe(all_queries, os.path.join(artifacts_dir, "mnist_query_history.csv"))

    # Main presentation plots.
    plot_accuracy_curves(all_history, os.path.join(artifacts_dir, "mnist_accuracy_curves.png"))
    plot_uncertainty_curves(all_history, os.path.join(artifacts_dir, "mnist_uncertainty_curves.png"))
    plot_runtime_curves(all_history, os.path.join(artifacts_dir, "mnist_runtime_curves.png"))
    plot_query_trajectory(all_queries, os.path.join(artifacts_dir, "mnist_query_trajectory.png"))

    print(f"\nSaved artifacts to: {artifacts_dir}")


if __name__ == "__main__":
    main()
