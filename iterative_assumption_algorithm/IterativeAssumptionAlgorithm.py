import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_ensemble import SequenceEnsemble

import os
import uuid
import torch
import pandas as pd
from torch.utils.data import DataLoader
from utils import save_module_unique, load_module, compute_n_u235

import glob
from train_ensemble import train_ensembles
import itertools
from load_data import HGRDataset, create_synthetic_csv

class Node:
    def __init__(self, score, D_t : list[str], ensemble : SequenceEnsemble, save_dir = "EnsembleModules",
                 query=None, depth=0, used_queries=None):
        self.score = float(score)
        self.D_t = list(D_t) if isinstance(D_t, list) else D_t
        self.children = []
        self.labeled = False
        self.label = None
        self.query = query
        self.depth = depth
        self.used_queries = [] if used_queries is None else list(used_queries)
        self.ensemble_path = save_module_unique(ensemble, save_dir, "ensemble", save_state_dict=False)


    def add_child(self, node):
        self.children.append(node)

    def load_ensemble(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        return load_module(None, self.ensemble_path,map_location=device, expects_state_dict=False)


def r_hat(ensemble : SequenceEnsemble, grid : list):
    ensemble.eval()
    grid_vals = []
    for i, x in enumerate(grid):
        print(f"\r\t\t iteration {i}/{len(grid)}",end="")
        val = ensemble.average_over_selection(*x,MAX_ITERS=1)
        grid_vals.append(val)
    print("\n")
    return sum(grid_vals)/len(grid_vals), grid_vals


def policy(ensemble : SequenceEnsemble, grid : list, grid_vals : list, n_queries =3, lmbda=1, forbidden_queries=None):
    ensemble.eval()
    assert n_queries >= 1
    best_queries = []
    chosen_idxs = set()
    forbidden_queries = [] if forbidden_queries is None else list(forbidden_queries)
    working_grid_vals = list(grid_vals)

    for i in range(min(n_queries, len(grid))):
        print(f"\t iteration {i}/{n_queries}",end="")
        adjusted_vals = working_grid_vals[:]

        for i in range(len(grid)):
            candidate = grid[i]

            if i in chosen_idxs or candidate in forbidden_queries:
                adjusted_vals[i] = -float("inf")
                continue

            for best_query in best_queries:
                adjusted_vals[i] += ensemble.compute_distance(candidate, best_query, MAX_ITERS=1, simple=True)* lmbda

        best_idx = max(range(len(grid)), key=lambda i: adjusted_vals[i])

        if adjusted_vals[best_idx] == -float("inf"):
            break

        best_query = grid[best_idx]
        best_queries.append(best_query)
        chosen_idxs.add(best_idx)

    return best_queries



# Make sure these are properly imported in your actual script
# from load_data import HGRDataset, create_synthetic_csv
# from train_ensemble import SequenceEnsemble

def take_environment_step(D_t, new_query, old_ensemble, ensemble_training_function,
                          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Simulates an environment step in the active learning/RL setting.

    Args:
        D_t: Current dataset representation (can be a list of paths or a directory path).
        new_query: Tuple corresponding to (u_percent, IV, density, n_u_235, t).
        old_ensemble: The currently trained SequenceEnsemble instance.
        ensemble_training_function: Function that takes the updated D_t and returns a newly trained ensemble.
        device: Torch device string or object.
    """
    old_ensemble.eval()

    # 1. Parse the new_query (matches the format found in compute_distance)
    u_percent, IV,density, n_u_235, t = new_query

    # 2. Create a temporary synthetic CSV.
    # This automatically writes rows for every static position in VEHICLE_STATIC_POSITIONS[IV].
    # We use MAX_LEN=72 so the CSV has 80 columns total (8 feature + 72 sequence cols) matching load_data().
    temp_csv_name = f"pseudo_query_{uuid.uuid4().hex[:8]}.csv"
    create_synthetic_csv(temp_csv_name, u_percent, IV, density, n_u_235, t, MAX_LEN=72)

    # 3. Load the dataset so we get the correctly scaled and position-encoded tensors
    dataset = HGRDataset(
        [temp_csv_name],
        x_mean=old_ensemble.x_mean,
        x_std=old_ensemble.x_std,
        y_mean=old_ensemble.y_mean,
        y_std=old_ensemble.y_std
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_mean_preds = []

    # 4. Predict time series across all reactor positions
    with torch.no_grad():
        for X_batch, t_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            t_batch = t_batch.to(device)

            # preds shape: [num_ensemble_members, batch_size, T]
            preds = old_ensemble.member_predictions(X_batch, t_batch, T=72)

            # Average across ensemble members to generate the pseudo-label: shape [batch_size, T]
            mean_preds = preds.mean(dim=0)

            # Un-normalize back to raw scales so the CSV contains proper label formats
            # Note: y_std and y_mean are stored as unsqueezed scalars in your dataset/ensemble
            y_std = old_ensemble.y_std.to(device)
            y_mean = old_ensemble.y_mean.to(device)
            mean_preds_unnorm = mean_preds * y_std + y_mean

            all_mean_preds.append(mean_preds_unnorm.cpu())

    # Concatenate all unnormalized predictions (matches the row length of our temp CSV)
    all_mean_preds = torch.cat(all_mean_preds, dim=0).numpy()

    # 5. Overwrite the zeroed time series in the CSV with the pseudo-labels
    df = pd.read_csv(temp_csv_name)

    # In your encode function, time-series targets correspond to original columns 8:80 (72 columns)
    time_series_cols = df.columns[8:8 + 72]

    # Verify dimensions align before assigning
    assert len(df) == len(all_mean_preds), "Length mismatch between CSV rows and dataloader output"
    df.loc[:, time_series_cols] = np.rint(np.squeeze(all_mean_preds[:, :len(time_series_cols)]))

    # 6. Save the properly pseudo-labeled data into D_t's tracking structure
    if isinstance(D_t, list):
        # D_t is a list of file paths
        df.to_csv(temp_csv_name, index=False)  # update file in place
        D_t.append(temp_csv_name)
    else:
        # Assuming D_t is a directory string (e.g., 'fuel/')
        final_csv_path = os.path.join(D_t, temp_csv_name)
        df.to_csv(final_csv_path, index=False)
        if os.path.exists(temp_csv_name):
            os.remove(temp_csv_name)

    # 7. Step the environment by training the new ensemble on the augmented dataset
    new_ensemble = ensemble_training_function(D_t)

    return D_t, new_ensemble


def subtree_value(node : Node, mode : str = "terminal") -> float:
    if not node.children:
        return node.score

    child_vals = [subtree_value(child, mode=mode) for child in node.children]

    if mode == "terminal":
        return max(node.score, max(child_vals))
    elif mode == "cumulative":
        return node.score + max(child_vals)
    else:
        raise ValueError(f"Unknown backup mode: {mode}")


def best_root_child(root : Node, mode : str = "terminal"):
    if not root.children:
        return None
    return max(root.children, key=lambda child: subtree_value(child, mode=mode))


def best_root_children(root : Node, n : int = 1, mode : str = "terminal"):
    if not root.children:
        return []
    ranked_children = sorted(root.children, key=lambda child: subtree_value(child, mode=mode), reverse=True)
    return ranked_children[:n]


def best_root_query(root : Node, mode : str = "terminal"):
    best_child = best_root_child(root, mode=mode)
    return None if best_child is None else best_child.query


def next_state_select(root : Node, mode : str = "terminal") -> Node:
    if not root.children:
        return root

    node = best_root_child(root, mode=mode)

    while node.children:
        node = max(node.children, key=lambda child: subtree_value(child, mode=mode))

    return node


def lookahead_choice_algo(D_0 : list[str], ensemble_training_function, num_search_iters : int, grid : list,
                          n_return_queries : int, n_policy_queries=3, lmbda =  1,
                          backup_mode = "terminal",
                          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ):

    D_t = list(D_0)
    print("making first ensemble")
    ensemble = ensemble_training_function(D_t)
    score, grid_vals  = r_hat(ensemble, grid)
    root = Node(score, D_t, ensemble)

    current_node = root
    root.labeled = True

    for i in range(num_search_iters):
        print(f"\nBeginning Search iter {i}")
        current_ensemble = current_node.load_ensemble(device)
        print("\tloaded ensemble, beginning computation of r_hat")
        score, grid_vals = r_hat(current_ensemble, grid)
        print("\tcomputing policy:")
        best_queries = policy(
            current_ensemble,
            grid,
            grid_vals,
            n_queries=n_policy_queries,
            lmbda=lmbda,
            forbidden_queries=current_node.used_queries
        )

        if len(best_queries) == 0:
            break

        if not current_node.children:
            print("computing children:")
            for query in best_queries:
                D_t_plus_one = list(current_node.D_t) if isinstance(current_node.D_t, list) else current_node.D_t
                print("\t\tTaking environment step on child")
                D_t_plus_one, new_ensemble = take_environment_step(
                    D_t_plus_one,
                    query,
                    current_ensemble,
                    ensemble_training_function,
                    device = device
                )
                print("\t\t Computing r_hat on child")
                score, new_grid_vals = r_hat(new_ensemble, grid)
                print("\t\t adding child to current node")
                current_node.add_child(
                    Node(
                        score,
                        D_t_plus_one,
                        new_ensemble,
                        query=query,
                        depth=current_node.depth + 1,
                        used_queries=current_node.used_queries + [query]
                    )
                )

            current_node.labeled = True

        print("\t selecting next node")
        current_node = next_state_select(root, mode=backup_mode)
    print("Ranking root children")
    ranked_root_children = best_root_children(root, n=n_return_queries, mode=backup_mode)
    print("Taking winner")
    winner = ranked_root_children[0] if len(ranked_root_children) > 0 else None
    winner_query = None if winner is None else winner.query
    print("Ending Algo")
    return winner, winner_query, root, ranked_root_children

def train(D_t, path = "ensembles"):
    x_mean = torch.Tensor([[0.5998, 0.3474, 2.5385, 2.0769, 2.0000, 0.5435, 0.1667, 0.1667, 0.1667,
         0.1667, 0.1667, 0.1667, 0.2849, 0.4783]])
    x_std = torch.Tensor([[0.1667, 0.4440, 0.8436, 0.7305, 0.8174, 0.4015, 0.3731, 0.3731, 0.3731,
         0.3731, 0.3731, 0.3731, 0.3636, 0.3135]])
    y_mean =  torch.Tensor([171.1579])
    y_std = torch.Tensor([105.260])
    train_ensembles(file_paths=D_t, device=torch.device('cpu'), overwrite_file=True, T=30, per_fuel_ensembles=1)
    return SequenceEnsemble(path,x_mean, x_std, y_mean, y_std)


if __name__ == "__main__":

    file_path = "fuel_backup/"

    grid_resolution = 1

    u_precent_grid = [.35 + i*(5-.35)/grid_resolution for i in range(grid_resolution+1)]
    u_density_grid = [.8 + i*(1-.8)/grid_resolution for i in range(grid_resolution+1)]
    full_density_grid = [9000 + i*(16000-9000)/grid_resolution for i in range(grid_resolution+1)]
    vehicle_grid = [1,2]
    fuel_schedule_grid = [0,1]

    grid = itertools.product(u_precent_grid, u_density_grid, vehicle_grid, full_density_grid, fuel_schedule_grid)

    grid = [(u_percent, vehicle, u_density, compute_n_u235(u_percent, full_density,heavy_metal_fraction=u_density), fuel_schedule)
    for (u_percent, u_density, vehicle, full_density,fuel_schedule ) in grid]


    lookahead_choice_algo(glob.glob(file_path + "*.csv"),train,10,grid, n_return_queries=3)