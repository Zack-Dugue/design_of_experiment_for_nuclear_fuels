import copy
import os
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from mnist_plotting_utils import (
    ensure_dir,
    save_dataframe,
    plot_accuracy_curves,
    plot_uncertainty_curves,
    plot_runtime_curves,
    plot_query_trajectory,
)

# ============================================================
# Global performance knobs.
# ============================================================

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ============================================================
# Small CNN used by every ensemble member.
# ============================================================

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """
    LayerNorm for NCHW conv features.

    Usage:
        nn.Conv2d(32, 64, 3, padding=1),
        LayerNorm2d(64),
        nn.ReLU()

    This normalizes each sample independently across C,H,W.
    Shape: [N, C, H, W]
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            # Per-channel affine parameters, like BatchNorm2d.
            # Broadcasts over N,H,W.
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"LayerNorm2d expected 4D input [N,C,H,W], got shape {tuple(x.shape)}")

        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x = x * self.weight + self.bias

        return x

class SmallMNISTNet(nn.Module):
    """
    Small MNIST convnet for repeated retraining / active learning.

    Input:  [N, 1, 28, 28]
    Output: [N, 10]
    """

    def __init__(self, feature_dim: int = 64):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, 32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, 64)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 7 * 7, feature_dim)
        self.fc2 = nn.Linear(feature_dim, 10)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.norm1(self.conv1(x))))  # [N, 32, 14, 14]
        x = self.pool(F.relu(self.norm2(self.conv2(x))))  # [N, 64, 7, 7]
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))                           # [N, feature_dim]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.fc2(features)
        return logits


# ============================================================
# Tensor-backed MNIST state.
# ============================================================

@dataclass
class RealState:
    labeled_true_indices: List[int]
    unlabeled_pool_indices: List[int]


@dataclass
class SimulatedState:
    labeled_true_indices: List[int]
    unlabeled_pool_indices: List[int]
    pseudo_logits_by_index: Dict[int, torch.Tensor] = field(default_factory=dict)


class MixedMNISTTensorDataset(Dataset):
    """
    Tensor-backed dataset that mixes:
    - truly labeled examples      -> cross entropy
    - pseudo-labeled examples     -> MSE to target logits

    Using tensors instead of torchvision Dataset objects makes it much easier
    to share data across worker processes and avoids repeated transform / PIL
    overhead in hot loops.
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        true_indices: Sequence[int],
        pseudo_logits_by_index: Dict[int, torch.Tensor],
        sampled_indices: Optional[Sequence[int]] = None,
    ):
        self.images = images
        self.labels = labels
        self.true_indices = list(int(x) for x in true_indices)
        self.pseudo_logits_by_index = {
            int(k): v.detach().cpu().clone().float()
            for k, v in pseudo_logits_by_index.items()
        }
        if sampled_indices is None:
            self.indices = self.true_indices + list(self.pseudo_logits_by_index.keys())
        else:
            self.indices = list(int(x) for x in sampled_indices)
        self._pseudo_index_set = set(self.pseudo_logits_by_index.keys())

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = int(self.indices[idx])
        image = self.images[base_idx]
        true_label = int(self.labels[base_idx].item())

        if base_idx in self._pseudo_index_set:
            target_logits = self.pseudo_logits_by_index[base_idx]
            use_pseudo = 1
        else:
            target_logits = torch.zeros(10, dtype=torch.float32)
            use_pseudo = 0

        return (
            image,
            torch.tensor(true_label, dtype=torch.long),
            target_logits,
            torch.tensor(use_pseudo, dtype=torch.long),
        )


class IndexTensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, indices: Sequence[int]):
        self.images = images
        self.labels = labels
        self.indices = list(int(x) for x in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = int(self.indices[idx])
        return self.images[base_idx], self.labels[base_idx]


# ============================================================
# Ensemble wrapper.
# ============================================================

class ClassifierEnsemble(nn.Module):
    def __init__(self, members: List[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    @property
    def num_members(self) -> int:
        return len(self.members)

    @torch.inference_mode()
    def member_logits(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([m(x).unsqueeze(0) for m in self.members], dim=0)

    @torch.inference_mode()
    def average_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.member_logits(x).mean(dim=0)

    @torch.inference_mode()
    def uncertainty_scores(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.member_logits(x)
        return logits.var(dim=0).mean(dim=1)

    @torch.inference_mode()
    def average_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = [m.extract_features(x).unsqueeze(0) for m in self.members]
        return torch.cat(feats, dim=0).mean(dim=0)

    def to_cpu_inplace(self):
        self.cpu()
        return self

    def to_device_inplace(self, device: torch.device):
        self.to(device)
        return self


# ============================================================
# Generic helpers.
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bootstrap_sample_indices(base_indices: Sequence[int], rng: np.random.Generator) -> List[int]:
    base_indices = np.asarray(list(base_indices), dtype=np.int64)
    if len(base_indices) == 0:
        return []
    return rng.choice(base_indices, size=len(base_indices), replace=True).tolist()


@torch.inference_mode()
def evaluate_ensemble_accuracy(
    ensemble: ClassifierEnsemble,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
    dataloader_workers: int = 0,
) -> float:
    ensemble.eval()
    ensemble.to_device_inplace(device)

    dataset = IndexTensorDataset(test_images, test_labels, list(range(len(test_labels))))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = ensemble.average_logits(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


# ============================================================
# Parallel member training infrastructure.
# ============================================================

_WORKER_TRAIN_IMAGES = None
_WORKER_TRAIN_LABELS = None
_WORKER_DEVICE = None
_WORKER_DATALOADER_WORKERS = 0
_WORKER_AMP_ENABLED = False


def _parallel_worker_init(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    fixed_device_id: Optional[int],
    dataloader_workers: int,
    amp_enabled: bool,
):
    global _WORKER_TRAIN_IMAGES, _WORKER_TRAIN_LABELS
    global _WORKER_DEVICE, _WORKER_DATALOADER_WORKERS, _WORKER_AMP_ENABLED

    _WORKER_TRAIN_IMAGES = train_images
    _WORKER_TRAIN_LABELS = train_labels
    _WORKER_DATALOADER_WORKERS = int(dataloader_workers)
    _WORKER_AMP_ENABLED = bool(amp_enabled)

    if fixed_device_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(fixed_device_id)
        _WORKER_DEVICE = torch.device(f"cuda:{fixed_device_id}")
    else:
        _WORKER_DEVICE = torch.device("cpu")

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _make_grad_scaler(device: torch.device, enabled: bool):
    if not enabled:
        try:
            return torch.amp.GradScaler(device.type, enabled=False)
        except Exception:
            if device.type == "cuda":
                return torch.cuda.amp.GradScaler(enabled=False)
            return None

    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except Exception:
        if device.type == "cuda":
            return torch.cuda.amp.GradScaler(enabled=enabled)
        return None


def _train_member_worker(task: Dict) -> Dict:
    """
    Train ONE ensemble member in a dedicated worker process.

    IMPORTANT:
    This is wrapped in try/except so Python-side worker failures print their
    traceback instead of only surfacing as BrokenProcessPool in the parent.
    """
    try:
        assert _WORKER_TRAIN_IMAGES is not None
        assert _WORKER_TRAIN_LABELS is not None
        assert _WORKER_DEVICE is not None

        device = _WORKER_DEVICE
        use_amp = (_WORKER_AMP_ENABLED and device.type == "cuda")

        dataset = MixedMNISTTensorDataset(
            images=_WORKER_TRAIN_IMAGES,
            labels=_WORKER_TRAIN_LABELS,
            true_indices=task["true_indices"],
            pseudo_logits_by_index=task["pseudo_logits_by_index"],
            sampled_indices=task["bootstrap_indices"],
        )

        loader = DataLoader(
            dataset,
            batch_size=task["batch_size"],
            shuffle=True,
            num_workers=_WORKER_DATALOADER_WORKERS,
            pin_memory=(device.type == "cuda"),
            persistent_workers=False,
        )

        model = SmallMNISTNet(feature_dim=task["feature_dim"])
        if task["init_state_dict"] is not None:
            model.load_state_dict(task["init_state_dict"])
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=task["lr"])
        scaler = _make_grad_scaler(device, enabled=use_amp)

        total_true_ce = 0.0
        total_pseudo_mse = 0.0
        total_batches = 0

        model.train()
        for _ in range(task["epochs"]):
            for images, true_labels, target_logits, use_pseudo in loader:
                images = images.to(device, non_blocking=True)
                true_labels = true_labels.to(device, non_blocking=True)
                target_logits = target_logits.to(device, non_blocking=True)
                use_pseudo = use_pseudo.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(images)
                    loss = torch.zeros((), device=device)

                    true_mask = (use_pseudo == 0)
                    if true_mask.any():
                        ce_loss = F.cross_entropy(logits[true_mask], true_labels[true_mask])
                        loss = loss + ce_loss
                        total_true_ce += float(ce_loss.detach().item())

                    pseudo_mask = (use_pseudo == 1)
                    if pseudo_mask.any():
                        mse_loss = F.mse_loss(logits[pseudo_mask], target_logits[pseudo_mask])
                        loss = loss + task["pseudo_loss_weight"] * mse_loss
                        total_pseudo_mse += float(mse_loss.detach().item())

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_batches += 1

        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        model.cpu()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

        return {
            "state_dict": state_dict,
            "mean_true_ce": total_true_ce / max(total_batches, 1),
            "mean_pseudo_mse": total_pseudo_mse / max(total_batches, 1),
        }

    except Exception as e:
        print(f"[train worker failed on device={_WORKER_DEVICE}] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise


def _train_member_local(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
    dataloader_workers: int,
    amp_enabled: bool,
    task: Dict,
) -> Dict:
    global _WORKER_TRAIN_IMAGES, _WORKER_TRAIN_LABELS
    global _WORKER_DEVICE, _WORKER_DATALOADER_WORKERS, _WORKER_AMP_ENABLED

    _WORKER_TRAIN_IMAGES = train_images
    _WORKER_TRAIN_LABELS = train_labels
    _WORKER_DEVICE = device
    _WORKER_DATALOADER_WORKERS = dataloader_workers
    _WORKER_AMP_ENABLED = amp_enabled

    return _train_member_worker(task)


class ParallelMemberTrainer:
    """
    Persistent trainer with one worker process per GPU.
    """

    def __init__(
        self,
        train_images: torch.Tensor,
        train_labels: torch.Tensor,
        device_ids: Sequence[int],
        dataloader_workers_per_worker: int = 0,
        amp_enabled: bool = True,
    ):
        self.train_images = train_images
        self.train_labels = train_labels
        self.device_ids = list(int(x) for x in device_ids)
        self.dataloader_workers_per_worker = int(dataloader_workers_per_worker)
        self.amp_enabled = bool(amp_enabled and torch.cuda.is_available())
        self.executors: List[ProcessPoolExecutor] = []
        self._build_executors()

    def _build_executors(self):
        self.shutdown()
        self.executors = []

        if len(self.device_ids) > 0:
            ctx = mp.get_context("spawn")
            for device_id in self.device_ids:
                executor = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=ctx,
                    initializer=_parallel_worker_init,
                    initargs=(
                        self.train_images,
                        self.train_labels,
                        device_id,
                        self.dataloader_workers_per_worker,
                        self.amp_enabled,
                    ),
                )
                self.executors.append(executor)

    def shutdown(self):
        for executor in self.executors:
            try:
                executor.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass
        self.executors = []

    def train_ensemble(
        self,
        true_indices: Sequence[int],
        pseudo_logits_by_index: Dict[int, torch.Tensor],
        test_images: torch.Tensor,
        test_labels: torch.Tensor,
        primary_device: torch.device,
        num_members: int = 3,
        epochs: int = 3,
        batch_size: int = 64,
        lr: float = 1e-3,
        pseudo_loss_weight: float = 1.0,
        seed: int = 0,
        feature_dim: int = 64,
        warm_start_ensemble: Optional[ClassifierEnsemble] = None,
        eval_batch_size: int = 512,
        eval_dataloader_workers: int = 0,
    ) -> Tuple[ClassifierEnsemble, Dict[str, float]]:
        members = []
        member_metrics = []

        rng = np.random.default_rng(seed)
        base_indices = list(int(x) for x in true_indices) + [int(x) for x in pseudo_logits_by_index.keys()]

        init_state_dicts = None
        if warm_start_ensemble is not None and warm_start_ensemble.num_members >= num_members:
            init_state_dicts = [
                {k: v.detach().cpu().clone() for k, v in warm_start_ensemble.members[i].state_dict().items()}
                for i in range(num_members)
            ]

        tasks = []
        for member_id in range(num_members):
            tasks.append({
                "true_indices": list(int(x) for x in true_indices),
                "pseudo_logits_by_index": {
                    int(k): v.detach().cpu().clone().float() for k, v in pseudo_logits_by_index.items()
                },
                "bootstrap_indices": bootstrap_sample_indices(base_indices, rng),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "lr": float(lr),
                "pseudo_loss_weight": float(pseudo_loss_weight),
                "feature_dim": int(feature_dim),
                "init_state_dict": None if init_state_dicts is None else init_state_dicts[member_id],
            })

        parallel_available = (len(self.executors) > 0 and primary_device.type == "cuda")

        if parallel_available and num_members > 1:
            try:
                futures = []
                for member_id, task in enumerate(tasks):
                    executor = self.executors[member_id % len(self.executors)]
                    futures.append(executor.submit(_train_member_worker, task))
                results = [future.result() for future in futures]

            except BrokenProcessPool:
                print("[ParallelMemberTrainer] BrokenProcessPool detected. Rebuilding executors and falling back to sequential local training for this round.", flush=True)
                self._build_executors()
                results = []
                for task in tasks:
                    results.append(
                        _train_member_local(
                            train_images=self.train_images,
                            train_labels=self.train_labels,
                            device=primary_device,
                            dataloader_workers=0,
                            amp_enabled=self.amp_enabled,
                            task=task,
                        )
                    )

            except Exception:
                print("[ParallelMemberTrainer] Parallel member training failed. Rebuilding executors and falling back to sequential local training for this round.", flush=True)
                traceback.print_exc()
                self._build_executors()
                results = []
                for task in tasks:
                    results.append(
                        _train_member_local(
                            train_images=self.train_images,
                            train_labels=self.train_labels,
                            device=primary_device,
                            dataloader_workers=0,
                            amp_enabled=self.amp_enabled,
                            task=task,
                        )
                    )
        else:
            results = []
            for task in tasks:
                results.append(
                    _train_member_local(
                        train_images=self.train_images,
                        train_labels=self.train_labels,
                        device=primary_device,
                        dataloader_workers=0,
                        amp_enabled=self.amp_enabled,
                        task=task,
                    )
                )

        for result in results:
            model = SmallMNISTNet(feature_dim=feature_dim)
            model.load_state_dict(result["state_dict"])
            model.to(primary_device)
            model.eval()
            members.append(model)
            member_metrics.append({
                "mean_true_ce": result["mean_true_ce"],
                "mean_pseudo_mse": result["mean_pseudo_mse"],
            })

        ensemble = ClassifierEnsemble(members).to(primary_device)
        test_acc = evaluate_ensemble_accuracy(
            ensemble=ensemble,
            test_images=test_images,
            test_labels=test_labels,
            device=primary_device,
            batch_size=eval_batch_size,
            dataloader_workers=eval_dataloader_workers,
        )

        summary = {
            "test_accuracy": float(test_acc),
            "mean_true_ce": float(np.mean([m["mean_true_ce"] for m in member_metrics])),
            "mean_pseudo_mse": float(np.mean([m["mean_pseudo_mse"] for m in member_metrics])),
        }
        return ensemble, summary


# ============================================================
# Scoring / selection helpers.
# ============================================================

@torch.inference_mode()
def compute_grid_scores(
    ensemble: ClassifierEnsemble,
    images: torch.Tensor,
    labels: torch.Tensor,
    candidate_indices: Sequence[int],
    device: torch.device,
    batch_size: int = 512,
    dataloader_workers: int = 0,
) -> Dict[int, float]:
    if len(candidate_indices) == 0:
        return {}

    ensemble.eval()
    ensemble.to_device_inplace(device)

    dataset = IndexTensorDataset(images, labels, candidate_indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    scores = {}
    ordered = list(int(x) for x in candidate_indices)
    cursor = 0

    for batch_images, _ in loader:
        batch_images = batch_images.to(device, non_blocking=True)
        batch_scores = ensemble.uncertainty_scores(batch_images).detach().cpu().tolist()
        for value in batch_scores:
            scores[ordered[cursor]] = float(value)
            cursor += 1

    return scores


@torch.inference_mode()
def compute_feature_embeddings(
    ensemble: ClassifierEnsemble,
    images: torch.Tensor,
    labels: torch.Tensor,
    candidate_indices: Sequence[int],
    device: torch.device,
    batch_size: int = 512,
    dataloader_workers: int = 0,
) -> Dict[int, torch.Tensor]:
    if len(candidate_indices) == 0:
        return {}

    ensemble.eval()
    ensemble.to_device_inplace(device)

    dataset = IndexTensorDataset(images, labels, candidate_indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    feats = {}
    ordered = list(int(x) for x in candidate_indices)
    cursor = 0

    for batch_images, _ in loader:
        batch_images = batch_images.to(device, non_blocking=True)
        batch_feats = ensemble.average_features(batch_images).detach().cpu()
        for row in batch_feats:
            feats[ordered[cursor]] = row.clone()
            cursor += 1

    return feats


def feature_distance(feature_a: torch.Tensor, feature_b: torch.Tensor) -> float:
    a = feature_a.float()
    b = feature_b.float()
    denom = (a.norm() * b.norm()).clamp_min(1e-8)
    cos_sim = float(torch.dot(a, b) / denom)
    return 1.0 - cos_sim


def r_hat_from_grid_scores(grid_scores: Dict[int, float]) -> float:
    if len(grid_scores) == 0:
        return 0.0
    return float(np.mean(list(grid_scores.values())))


def screen_top_uncertain(grid_scores: Dict[int, float], screen_size: int) -> List[int]:
    ranked = sorted(grid_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [int(idx) for idx, _ in ranked[:min(screen_size, len(ranked))]]


def policy_select_indices(
    grid_scores: Dict[int, float],
    feature_cache: Dict[int, torch.Tensor],
    n_queries: int = 3,
    lmbda: float = 1.0,
    forbidden_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    forbidden = set([] if forbidden_indices is None else list(int(x) for x in forbidden_indices))
    chosen: List[int] = []
    candidate_indices = list(feature_cache.keys())

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


@torch.inference_mode()
def pseudo_label_index(
    ensemble: ClassifierEnsemble,
    images: torch.Tensor,
    data_index: int,
    device: torch.device,
) -> torch.Tensor:
    ensemble.eval()
    ensemble.to_device_inplace(device)
    image = images[int(data_index)].unsqueeze(0).to(device)
    return ensemble.average_logits(image).squeeze(0).detach().cpu()


# ============================================================
# Tree / search helpers.
# ============================================================

class Node:
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
        self.ensemble = ensemble.to_cpu_inplace()
        self.grid_scores = {} if grid_scores is None else dict(grid_scores)
        self.query_index = query_index
        self.depth = depth
        self.used_indices = [] if used_indices is None else list(used_indices)
        self.children: List["Node"] = []

    def add_child(self, child: "Node"):
        self.children.append(child)

    def load_ensemble(self, device: torch.device) -> ClassifierEnsemble:
        return self.ensemble.to_device_inplace(device)


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


def simulate_environment_step(
    node: Node,
    chosen_index: int,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    primary_device: torch.device,
    trainer: ParallelMemberTrainer,
    num_members: int,
    train_epochs: int,
    batch_size: int,
    lr: float,
    pseudo_loss_weight: float,
    feature_dim: int,
    score_batch_size: int,
    score_dataloader_workers: int,
    seed: int,
) -> Tuple[SimulatedState, ClassifierEnsemble, Dict[int, float], Dict[str, float]]:
    parent_ensemble = node.load_ensemble(primary_device)
    pseudo_logits = pseudo_label_index(parent_ensemble, train_images, chosen_index, primary_device)

    new_pseudo_logits = dict(node.sim_state.pseudo_logits_by_index)
    new_pseudo_logits[int(chosen_index)] = pseudo_logits
    new_pool = [idx for idx in node.sim_state.unlabeled_pool_indices if idx != chosen_index]

    new_state = SimulatedState(
        labeled_true_indices=list(node.sim_state.labeled_true_indices),
        unlabeled_pool_indices=new_pool,
        pseudo_logits_by_index=new_pseudo_logits,
    )

    ensemble, train_metrics = trainer.train_ensemble(
        true_indices=new_state.labeled_true_indices,
        pseudo_logits_by_index=new_state.pseudo_logits_by_index,
        test_images=test_images,
        test_labels=test_labels,
        primary_device=primary_device,
        num_members=num_members,
        epochs=train_epochs,
        batch_size=batch_size,
        lr=lr,
        pseudo_loss_weight=pseudo_loss_weight,
        seed=seed,
        feature_dim=feature_dim,
        warm_start_ensemble=parent_ensemble,
        eval_batch_size=score_batch_size,
        eval_dataloader_workers=score_dataloader_workers,
    )

    grid_scores = compute_grid_scores(
        ensemble=ensemble,
        images=train_images,
        labels=train_labels,
        candidate_indices=new_pool,
        device=primary_device,
        batch_size=score_batch_size,
        dataloader_workers=score_dataloader_workers,
    )

    return new_state, ensemble, grid_scores, train_metrics


def choose_query_with_lookahead(
    real_state: RealState,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    primary_device: torch.device,
    trainer: ParallelMemberTrainer,
    root_ensemble: ClassifierEnsemble,
    root_grid_scores: Dict[int, float],
    num_search_iters: int = 3,
    n_policy_queries: int = 3,
    lmbda: float = 0.5,
    backup_mode: str = "terminal",
    num_members: int = 3,
    train_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    pseudo_loss_weight: float = 1.0,
    feature_dim: int = 64,
    score_batch_size: int = 512,
    score_dataloader_workers: int = 0,
    feature_screen_size: int = 512,
    seed: int = 0,
) -> Tuple[int, Dict[str, object]]:
    sim_state = SimulatedState(
        labeled_true_indices=list(real_state.labeled_true_indices),
        unlabeled_pool_indices=list(real_state.unlabeled_pool_indices),
        pseudo_logits_by_index={},
    )

    root = Node(
        score=r_hat_from_grid_scores(root_grid_scores),
        sim_state=sim_state,
        ensemble=copy.deepcopy(root_ensemble),
        grid_scores=root_grid_scores,
        query_index=None,
        depth=0,
        used_indices=[],
    )

    history = []
    current_node = root

    for search_iter in range(num_search_iters):
        iter_start = time.perf_counter()

        screened = screen_top_uncertain(current_node.grid_scores, feature_screen_size)
        current_ensemble = current_node.load_ensemble(primary_device)
        feature_cache = compute_feature_embeddings(
            ensemble=current_ensemble,
            images=train_images,
            labels=train_labels,
            candidate_indices=screened,
            device=primary_device,
            batch_size=score_batch_size,
            dataloader_workers=score_dataloader_workers,
        )

        candidate_indices = policy_select_indices(
            grid_scores=current_node.grid_scores,
            feature_cache=feature_cache,
            n_queries=n_policy_queries,
            lmbda=lmbda,
            forbidden_indices=current_node.used_indices,
        )

        expansion_summaries = []
        if not current_node.children:
            for local_rank, chosen_index in enumerate(candidate_indices, start=1):
                child_state, child_ensemble, child_grid_scores, child_train_metrics = simulate_environment_step(
                    node=current_node,
                    chosen_index=chosen_index,
                    train_images=train_images,
                    train_labels=train_labels,
                    test_images=test_images,
                    test_labels=test_labels,
                    primary_device=primary_device,
                    trainer=trainer,
                    num_members=num_members,
                    train_epochs=train_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    pseudo_loss_weight=pseudo_loss_weight,
                    feature_dim=feature_dim,
                    score_batch_size=score_batch_size,
                    score_dataloader_workers=score_dataloader_workers,
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
                    "query_index": int(chosen_index),
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
            "screen_size": len(screened),
            "expansions": expansion_summaries,
        })

    chosen_root = best_root_child(root, mode=backup_mode)
    if chosen_root is None:
        fallback = random.choice(real_state.unlabeled_pool_indices)
        return fallback, {"search_history": history, "root_score": float(root.score)}

    return int(chosen_root.query_index), {
        "search_history": history,
        "root_score": float(root.score),
        "chosen_root_subtree_value": float(subtree_value(chosen_root, mode=backup_mode)),
    }


# ============================================================
# Real-state training / baselines.
# ============================================================

def initialize_real_state(
    train_labels: torch.Tensor,
    initial_labeled_size: int,
    pool_size: int,
    seed: int,
) -> RealState:
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(train_labels))
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
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    primary_device: torch.device,
    trainer: ParallelMemberTrainer,
    num_members: int,
    train_epochs: int,
    batch_size: int,
    lr: float,
    feature_dim: int,
    score_batch_size: int,
    score_dataloader_workers: int,
    seed: int,
    warm_start_ensemble: Optional[ClassifierEnsemble] = None,
) -> Tuple[ClassifierEnsemble, Dict[int, float], Dict[str, float]]:
    ensemble, train_metrics = trainer.train_ensemble(
        true_indices=state.labeled_true_indices,
        pseudo_logits_by_index={},
        test_images=test_images,
        test_labels=test_labels,
        primary_device=primary_device,
        num_members=num_members,
        epochs=train_epochs,
        batch_size=batch_size,
        lr=lr,
        pseudo_loss_weight=1.0,
        seed=seed,
        feature_dim=feature_dim,
        warm_start_ensemble=warm_start_ensemble,
        eval_batch_size=score_batch_size,
        eval_dataloader_workers=score_dataloader_workers,
    )

    pool_scores = compute_grid_scores(
        ensemble=ensemble,
        images=train_images,
        labels=train_labels,
        candidate_indices=state.unlabeled_pool_indices,
        device=primary_device,
        batch_size=score_batch_size,
        dataloader_workers=score_dataloader_workers,
    )

    summary = dict(train_metrics)
    summary["pool_mean_uncertainty"] = r_hat_from_grid_scores(pool_scores)
    summary["pool_max_uncertainty"] = float(max(pool_scores.values())) if len(pool_scores) > 0 else 0.0
    return ensemble, pool_scores, summary


# ============================================================
# Outer active-learning loop.
# ============================================================

def build_mnist_tensors(data_root: str = "./data"):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    train_images = train_dataset.data.float().div_(255.0).unsqueeze(1).contiguous()
    train_labels = train_dataset.targets.long().contiguous()
    test_images = test_dataset.data.float().div_(255.0).unsqueeze(1).contiguous()
    test_labels = test_dataset.targets.long().contiguous()

    train_images.share_memory_()
    train_labels.share_memory_()
    test_images.share_memory_()
    test_labels.share_memory_()

    return train_images, train_labels, test_images, test_labels


def run_strategy_experiment(
    strategy_name: str,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    primary_device: torch.device,
    trainer: ParallelMemberTrainer,
    initial_labeled_size: int = 100,
    pool_size: int = 5000,
    acquisition_steps: int = 10,
    num_members: int = 4,
    train_epochs: int = 2,
    batch_size: int = 128,
    lr: float = 1e-3,
    lookahead_search_iters: int = 2,
    lookahead_branching: int = 3,
    lookahead_lambda: float = 0.35,
    feature_dim: int = 64,
    score_batch_size: int = 512,
    score_dataloader_workers: int = 0,
    feature_screen_size: int = 512,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    state = initialize_real_state(train_labels, initial_labeled_size, pool_size, seed)

    history_rows = []
    query_rows = []
    previous_real_ensemble = None

    for step in range(acquisition_steps + 1):
        outer_start = time.perf_counter()
        print(f"\rAcquisition Step : {step}", end="")

        ensemble, pool_scores, eval_metrics = fit_and_evaluate_real_state(
            state=state,
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
            primary_device=primary_device,
            trainer=trainer,
            num_members=num_members,
            train_epochs=train_epochs,
            batch_size=batch_size,
            lr=lr,
            feature_dim=feature_dim,
            score_batch_size=score_batch_size,
            score_dataloader_workers=score_dataloader_workers,
            seed=seed + 17 * step,
            warm_start_ensemble=previous_real_ensemble,
        )
        previous_real_ensemble = copy.deepcopy(ensemble).to_cpu_inplace()

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

        if step == acquisition_steps or len(state.unlabeled_pool_indices) == 0:
            row["outer_step_seconds"] = float(time.perf_counter() - outer_start)
            history_rows.append(row)
            break

        if strategy_name == "random":
            chosen_index = rng.choice(state.unlabeled_pool_indices)
        elif strategy_name == "uncertainty":
            chosen_index = max(pool_scores.keys(), key=lambda idx: pool_scores[idx])
        elif strategy_name == "lookahead":
            chosen_index, _ = choose_query_with_lookahead(
                real_state=state,
                train_images=train_images,
                train_labels=train_labels,
                test_images=test_images,
                test_labels=test_labels,
                primary_device=primary_device,
                trainer=trainer,
                root_ensemble=ensemble,
                root_grid_scores=pool_scores,
                num_search_iters=lookahead_search_iters,
                n_policy_queries=lookahead_branching,
                lmbda=lookahead_lambda,
                backup_mode="terminal",
                num_members=num_members,
                train_epochs=train_epochs,
                batch_size=batch_size,
                lr=lr,
                pseudo_loss_weight=1.0,
                feature_dim=feature_dim,
                score_batch_size=score_batch_size,
                score_dataloader_workers=score_dataloader_workers,
                feature_screen_size=feature_screen_size,
                seed=seed + 200 * step,
            )
        else:
            raise ValueError(f"Unknown strategy_name={strategy_name}")

        img = train_images[int(chosen_index)]
        true_label = int(train_labels[int(chosen_index)].item())
        query_rows.append({
            "strategy": strategy_name,
            "step": step,
            "chosen_index": int(chosen_index),
            "true_label": true_label,
            "uncertainty_at_choice": float(pool_scores.get(int(chosen_index), np.nan)),
            "pixel_mean": float(img.mean().item()),
            "pixel_std": float(img.std().item()),
        })

        row["chosen_index"] = int(chosen_index)
        row["chosen_label"] = true_label
        row["outer_step_seconds"] = float(time.perf_counter() - outer_start)
        history_rows.append(row)

        state = commit_true_label(state, int(chosen_index))

    print("\nAcquisition concluded")
    return pd.DataFrame(history_rows), pd.DataFrame(query_rows)


# ============================================================
# Script entrypoint.
# ============================================================

def main():
    set_seed(0)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        device_ids = []
        primary_device = torch.device("cpu")
    else:
        device_ids = list(range(min(8, available_gpus)))
        primary_device = torch.device(f"cuda:{device_ids[0]}")

    print(f"Using device_ids={device_ids} | primary_device={primary_device}")

    train_images, train_labels, test_images, test_labels = build_mnist_tensors()

    trainer = ParallelMemberTrainer(
        train_images=train_images,
        train_labels=train_labels,
        device_ids=device_ids,
        dataloader_workers_per_worker=0,
        amp_enabled=True,
    )

    artifacts_dir = "mnist_lookahead_parallel_artifacts"
    ensure_dir(artifacts_dir)

    strategy_histories = []
    strategy_queries = []

    try:
        for strategy in ["random", "uncertainty", "lookahead"]:
            print(f"\nRunning strategy: {strategy}")
            history_df, query_df = run_strategy_experiment(
                strategy_name=strategy,
                train_images=train_images,
                train_labels=train_labels,
                test_images=test_images,
                test_labels=test_labels,
                primary_device=primary_device,
                trainer=trainer,
                initial_labeled_size=100,
                pool_size=5000,
                acquisition_steps=25,
                num_members=max(1, min(4, len(device_ids) if len(device_ids) > 0 else 1)),
                train_epochs=2,
                batch_size=128,
                lr=1e-3,
                lookahead_search_iters=2,
                lookahead_branching=3,
                lookahead_lambda=0.35,
                feature_dim=64,
                score_batch_size=512,
                score_dataloader_workers=0,
                feature_screen_size=512,
                seed=0,
            )
            strategy_histories.append(history_df)
            strategy_queries.append(query_df)
    finally:
        trainer.shutdown()

    all_history = pd.concat(strategy_histories, ignore_index=True)
    all_queries = pd.concat(strategy_queries, ignore_index=True)

    save_dataframe(all_history, os.path.join(artifacts_dir, "mnist_strategy_history.csv"))
    save_dataframe(all_queries, os.path.join(artifacts_dir, "mnist_query_history.csv"))

    plot_accuracy_curves(all_history, os.path.join(artifacts_dir, "mnist_accuracy_curves.png"))
    plot_uncertainty_curves(all_history, os.path.join(artifacts_dir, "mnist_uncertainty_curves.png"))
    plot_runtime_curves(all_history, os.path.join(artifacts_dir, "mnist_runtime_curves.png"))
    plot_query_trajectory(all_queries, os.path.join(artifacts_dir, "mnist_query_trajectory.png"))

    print(f"\nSaved artifacts to: {artifacts_dir}")


if __name__ == "__main__":
    main()
