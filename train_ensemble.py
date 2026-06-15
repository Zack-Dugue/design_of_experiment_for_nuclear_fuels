import glob
import os
from copy import deepcopy

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from load_data import HGRDataset, create_synthetic_csv
from time_schedules import N_STATIC_COLUMNS
from model import StaticFeatureTransformer, StaticFeatureTCN


class MAELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(MAELoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_hat, y):
        return torch.mean(torch.abs(y[:, 1:] - y_hat[:, :-1]))


class SequenceEnsemble(nn.Module):
    def __init__(self, path, x_mean, x_std, y_mean, y_std, device=torch.device("cpu")):
        super(SequenceEnsemble, self).__init__()
        self.ensemble_list = nn.ModuleList([])
        self.mock_mode = False
        self.n = 5

        try:
            model = torch.load(os.path.join(path, "class_example.mdl"), map_location=device, weights_only=False)
            file_paths = glob.glob(os.path.join(path, "*.pth"))
            if len(file_paths) == 0:
                raise FileNotFoundError
            for single_path in file_paths:
                copy_model = deepcopy(model)
                state_dict = torch.load(open(single_path, "rb"), map_location=device)
                copy_model.load_state_dict(state_dict)
                copy_model.to(device)
                copy_model.eval()
                self.ensemble_list.append(copy_model)
            self.n = len(self.ensemble_list)
            print(f"Successfully loaded {self.n} models from {path}.")
        except Exception as e:
            print(f"Failed to load models ({e}).")
            assert False

        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.device = device

    @torch.no_grad()
    def member_predictions(self, x, t, T=None):
        """
        Return predictions from every ensemble member.

        If T is None and t is an explicit time matrix [B,T], infer T from t. This
        avoids hard-coding sequence length when the new time format has a
        different number of time columns.
        """
        if T is None:
            if torch.is_tensor(t) and t.ndim == 2:
                T = t.size(1)
            else:
                T = 64

        preds = []
        for model in self.ensemble_list:
            y = model.decode(x, t, T)
            preds.append(y.unsqueeze(0))
        return torch.cat(preds, dim=0)

    def forward(self, x, t, T=None):
        return self.member_predictions(x, t, T=T).variance(dim=0).mean()

    def average_over_selection(
        self,
        u_percent,
        IV,
        density,
        n_u_235,
        t,
        MAX_ITERS=50,
        path="tmp.csv",
        batch_size=32,
        target_length=72,
    ):
        create_synthetic_csv(path, u_percent, IV, density, n_u_235, t, MAX_LEN=target_length)
        print(f"average over selection x_mean = {self.x_mean}")
        averaging_dataset = HGRDataset(
            [path],
            x_mean=self.x_mean,
            x_std=self.x_std,
            y_mean=self.y_mean,
            y_std=self.y_std,
            target_length=target_length,
        )
        averaging_dataloader = torch.utils.data.DataLoader(averaging_dataset, batch_size=batch_size, shuffle=True)
        average_score = 0
        count = 0
        for (X, t_batch, y) in averaging_dataloader:
            X, t_batch, y = X.to(self.device), t_batch.to(self.device), y.to(self.device)
            count += 1
            if MAX_ITERS < count:
                break
            average_score += self.member_predictions(X, t_batch, T=y.size(1)).var(0).mean()
        return average_score / count

    @staticmethod
    def cheap_compute_distance(y_1, y_2):
        distance = ((y_1 - y_2) ** 2).mean()
        return distance

    def compute_distance(self, X_1, X_2, MAX_ITERS=50, path="tmp.csv", batch_size=32, simple=False, target_length=50):
        with torch.no_grad():
            u_percent, IV, density, n_u_235, t = X_1
            create_synthetic_csv(path, u_percent, IV, density, n_u_235, t, MAX_LEN=target_length)
            averaging_dataset_1 = HGRDataset(
                [path],
                x_mean=self.x_mean,
                x_std=self.x_std,
                y_mean=self.y_mean,
                y_std=self.y_std,
                target_length=target_length,
            )
            averaging_dataloader_1 = torch.utils.data.DataLoader(
                averaging_dataset_1, batch_size=batch_size, shuffle=False
            )

            u_percent, IV, density, n_u_235, t = X_2
            create_synthetic_csv(path, u_percent, IV, density, n_u_235, t, MAX_LEN=target_length)
            averaging_dataset_2 = HGRDataset(
                [path],
                x_mean=self.x_mean,
                x_std=self.x_std,
                y_mean=self.y_mean,
                y_std=self.y_std,
                target_length=target_length,
            )
            averaging_dataloader_2 = torch.utils.data.DataLoader(
                averaging_dataset_2, batch_size=batch_size, shuffle=False
            )

            distance = 0
            count = 0
            for ((x_1, t_1, y_1_true), (x_2, t_2, y_2_true)) in zip(
                averaging_dataloader_1,
                averaging_dataloader_2,
            ):
                x_1, t_1 = x_1.to(self.device), t_1.to(self.device)
                x_2, t_2 = x_2.to(self.device), t_2.to(self.device)
                if simple:
                    distance += ((x_1[:, :4] - x_2[:, :4]) ** 2).mean()
                else:
                    y_1 = self.member_predictions(x_1, t_1, T=y_1_true.size(1))
                    y_2 = self.member_predictions(x_2, t_2, T=y_2_true.size(1))
                    length = min(y_1.size(1), y_2.size(1))
                    distance += self.cheap_compute_distance(y_1[:, :length], y_2[:, :length])
                count += 1
                if MAX_ITERS < count:
                    break
            return distance / count


def get_num_time_columns(path, n_static_columns=N_STATIC_COLUMNS):
    """Return the number of time/target columns in a processed CSV."""
    df = pd.read_csv(path, nrows=1)
    n_time = len(df.columns) - int(n_static_columns)
    if n_time <= 0:
        raise ValueError(
            f"{path} has {len(df.columns)} columns, which leaves {n_time} time columns "
            f"after {n_static_columns} static columns."
        )
    return n_time


def summarize_time_lengths(file_paths, n_static_columns=N_STATIC_COLUMNS):
    """Return per-file time lengths and the global common evaluation length."""
    lengths = {path: get_num_time_columns(path, n_static_columns=n_static_columns) for path in file_paths}
    global_min = min(lengths.values())

    print("\nDetected time/target column counts:")
    for path, length in sorted(lengths.items(), key=lambda kv: (kv[1], kv[0])):
        marker = "  <-- global eval length" if length == global_min else ""
        print(f"  {length:4d} | {path}{marker}")

    print(f"\nGlobal held-out evaluation target length = {global_min}")
    return lengths, global_min


def _make_model(num_features):
    return StaticFeatureTransformer(num_features, 256, 4, 512, 8, 0)


def evaluate_model(model, data_loader, device=torch.device("cuda")):
    avg_loss = 0
    loss_fun = MAELoss()
    model.eval()
    with torch.no_grad():
        for (i, (x, t, y)) in enumerate(data_loader):
            x = x.to(device)
            t = t.to(device)
            y = y.to(device)
            y_hat = model(x, t, y).squeeze(-1)
            loss = loss_fun(y_hat, y)
            avg_loss += loss.item()

    if isinstance(data_loader.dataset, torch.utils.data.Subset):
        y_std = data_loader.dataset.dataset.y_std
    else:
        y_std = data_loader.dataset.y_std
    return avg_loss * y_std / len(data_loader)


def train_ensembles(
    save_path="ensembles/",
    file_paths=glob.glob("fule/" + "*.csv"),
    per_fuel_ensembles=5,
    device=torch.device("cuda"),
    T=300,
    overwrite_file=False,
    target_length=None,
):
    print(file_paths)
    print(
        f"Training {per_fuel_ensembles} models for each {len(file_paths)} held out fuel "
        f"for a total of {per_fuel_ensembles * len(file_paths)} models"
    )

    if len(file_paths) == 0:
        raise ValueError("train_ensembles received no CSV file paths.")

    os.makedirs(save_path, exist_ok=True)

    # Held-out extrapolation should be comparable across folds and must work for
    # every held-out file. Use the shortest available curve across the *full*
    # file list for extrapolation evaluation only.
    #
    # Training still uses as much data as each fold can support:
    #   - if target_length is None, HGRDataset infers the shortest curve among
    #     the training files for that fold;
    #   - if target_length is provided, it is used as the requested training length.
    time_lengths, global_eval_target_length = summarize_time_lengths(file_paths)

    if target_length is not None:
        requested = int(target_length)
        if requested > global_eval_target_length:
            print(
                f"Warning: requested target_length={requested}, but the global held-out "
                f"evaluation length is only {global_eval_target_length}. Training will use "
                f"the requested length where possible; held-out evaluation will use "
                f"{global_eval_target_length}."
            )

    # Build one dataset only to determine the model input dimension. Use the
    # global evaluation length so this probe is guaranteed to work for all files.
    feature_probe_dataset = HGRDataset(file_paths, target_length=global_eval_target_length)
    num_features = feature_probe_dataset.X.shape[1]
    print(f"Detected model input feature count: {num_features}")
    print(f"Detected probe target length: {feature_probe_dataset.y.shape[1]}")

    class_example = _make_model(num_features)
    torch.save(class_example, os.path.join(save_path, "class_example.mdl"))

    for fuel in range(len(file_paths)):
        print(f"Onto fuel: {fuel}")
        hold_one_out_file_paths = file_paths.copy()
        held_out_file_path = hold_one_out_file_paths.pop(fuel)

        # Training target length: keep as much training horizon as this fold allows.
        # If target_length is None, the dataset uses the shortest training file in
        # this fold, which may be longer than the global held-out eval length.
        dataset = HGRDataset(hold_one_out_file_paths, target_length=target_length)
        fold_target_length = dataset.y.size(1)

        if dataset.time_values.shape[1] != dataset.y.shape[1]:
            raise ValueError(
                f"Fold {fuel} has mismatched t/y lengths: "
                f"t={dataset.time_values.shape[1]}, y={dataset.y.shape[1]}"
            )
        if dataset.X.shape[1] != num_features:
            raise ValueError(
                f"Fold {fuel} changed the feature count from {num_features} to {dataset.X.shape[1]}. "
                "This should not happen with fixed one-hot categories."
            )

        held_out_available_length = time_lengths[held_out_file_path]
        eval_target_length = min(global_eval_target_length, held_out_available_length, fold_target_length)

        print(
            f"Fold target length = {fold_target_length} | "
            f"held-out available length = {held_out_available_length} | "
            f"held-out eval length = {eval_target_length}"
        )

        for i in range(per_fuel_ensembles):
            print(f"\nGenerating Ensemble : {fuel}-{i}")
            save_file = f"{save_path}/{fuel}-{i}.pth"

            if os.path.isfile(save_file):
                print("This model has already been trained.")
                if overwrite_file:
                    os.remove(save_file)
                else:
                    continue

            train_set, interpolation_val_set = torch.utils.data.random_split(dataset, (0.5, 0.5))
            train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

            model = _make_model(num_features)
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0005)
            loss_fun = torch.nn.MSELoss(reduction="mean")
            eval_loss_fun = MAELoss(reduction="mean")
            lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)

            for epoch in range(T):
                model.train()
                avg_train_loss = 0
                for (j, (x, t, y)) in enumerate(train_loader):
                    x = x.to(device)
                    t = t.to(device)
                    y = y.to(device)

                    if t.ndim == 2 and t.size(1) != y.size(1):
                        raise ValueError(
                            f"Batch t/y mismatch before model call: t={tuple(t.shape)}, y={tuple(y.shape)}"
                        )

                    optimizer.zero_grad()
                    y_hat = model(x, t, y).squeeze(-1)
                    loss = loss_fun(y_hat[:, :-1], y[:, 1:])
                    loss.backward()
                    optimizer.step()
                    avg_train_loss += eval_loss_fun(y_hat, y).item()

                avg_train_loss = (avg_train_loss * train_loader.dataset.dataset.y_std) / len(train_loader)
                lr_schedule.step()
                if epoch == T - 1 or epoch % 5 == 0:
                    print(f"epoch = {epoch} : train_loss = {avg_train_loss} \r", end="")

            extrapolation_val_set = HGRDataset(
                [held_out_file_path],
                x_mean=dataset.x_mean,
                x_std=dataset.x_std,
                y_mean=dataset.y_mean,
                y_std=dataset.y_std,
                target_length=eval_target_length,
            )
            extrapolation_val_loader = DataLoader(extrapolation_val_set, batch_size=16)
            extrapolation_loss = evaluate_model(model, extrapolation_val_loader, device=device)

            interpolation_val_loader = DataLoader(interpolation_val_set, batch_size=16)
            interpolation_loss = evaluate_model(model, interpolation_val_loader, device=device)

            print(
                f"Model complete: Final Training loss={avg_train_loss}, "
                f"Final Interpolation Loss = {interpolation_loss}, "
                f"Final Extrapolation Loss = {extrapolation_loss}"
            )
            torch.save(model.state_dict(), save_file)


if __name__ == "__main__":
    train_ensembles(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        file_paths=glob.glob("HGR_fuel/" + "*.csv"),
        per_fuel_ensembles=3,
        T=200,
    )
