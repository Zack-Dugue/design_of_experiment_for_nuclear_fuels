"""
Find high-exploration candidate queries for the nuclear-fuels HGR ensemble.

This version is compatible with the generalized time-schedule setup where the
schedule identifier is an integer, not a boolean. Add new schedules in
`time_schedules.py`, then this script will automatically include them in the
Optuna search space.

Important tuple convention used throughout this file:
    query = (u_percent, IV, density, n_u_235, fuel_schedule)
which matches SequenceEnsemble.average_over_selection(...) and
SequenceEnsemble.compute_distance(...).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import csv

import optuna
import torch

from train_ensemble import SequenceEnsemble
from utils import compute_n_u235

try:
    # Added by the generalized time-schedule refactor.
    from time_schedules import TIME_SCHEDULE_GAPS
except Exception:
    # Backward-compatible fallback for the original two-schedule repo.
    TIME_SCHEDULE_GAPS = {
        0: [0.001, 0.999, 4, 5, 5, 5, 5],
        1: [0.001, 0.999, 2, 2, 5, 5, 5, 2, 2, 2],
    }


def available_schedule_ids() -> list[int]:
    """Return sorted integer schedule IDs that Optuna is allowed to choose."""
    return sorted(int(k) for k in TIME_SCHEDULE_GAPS.keys())


def make_objective(
    sequence_ensemble: SequenceEnsemble,
    prior_points: Sequence[tuple[float, int, float, float, int]],
    gamma: float = 0.0,
    schedule_ids: Sequence[int] | None = None,
):
    """
    Build an Optuna objective for selecting high-uncertainty candidate queries.

    The base score is ensemble predictive variance, computed by
    SequenceEnsemble.average_over_selection(...).

    If prior_points is non-empty and gamma > 0, we add a diversity term based on
    the minimum distance to the already-selected points. This makes greedy batch
    selection less likely to pick duplicates/nearly duplicates.

    Note: the old version accidentally overwrote the uncertainty score with the
    distance score inside the prior_points loop. This version combines them.
    """
    if schedule_ids is None:
        schedule_ids = available_schedule_ids()
    schedule_ids = list(schedule_ids)

    if not schedule_ids:
        raise ValueError("No fuel schedules are available. Check time_schedules.py.")

    def objective(trial: optuna.Trial) -> float:
        u_percent = trial.suggest_float("u_percent", 0.35, 5.0)
        u_density = trial.suggest_float("u_density", 0.8, 1.0)
        density = trial.suggest_float("density", 9000.0, 16000.0)
        IV = trial.suggest_int("IV", 1, 2)
        fuel_schedule = trial.suggest_categorical("fuel_schedule", schedule_ids)

        n_u_235 = compute_n_u235(
            u_percent,
            density,
            heavy_metal_fraction=u_density,
        )

        # Match SequenceEnsemble.average_over_selection signature:
        #   (u_percent, IV, density, n_u_235, t)
        uncertainty_score = sequence_ensemble.average_over_selection(
            u_percent,
            IV,
            density,
            n_u_235,
            int(fuel_schedule),
        )

        # Convert Tensor scores to plain floats for Optuna.
        if torch.is_tensor(uncertainty_score):
            uncertainty_score = uncertainty_score.detach().cpu().item()
        else:
            uncertainty_score = float(uncertainty_score)

        diversity_score = 0.0
        candidate = (u_percent, IV, density, n_u_235, int(fuel_schedule))
        if prior_points and gamma != 0:
            distances = []
            for prior_point in prior_points:
                d = sequence_ensemble.compute_distance(candidate, prior_point)
                if torch.is_tensor(d):
                    d = d.detach().cpu().item()
                distances.append(float(d))
            # Use minimum distance to encourage separation from the closest
            # already-selected query.
            diversity_score = min(distances)

        return uncertainty_score + gamma * diversity_score

    return objective


def find_best_queries(
    sequence_ensemble: SequenceEnsemble,
    write_dir: str | Path = "out",
    n_trials: int = 100,
    num_samples: int = 3,
    gamma: float = 0.0,
    schedule_ids: Sequence[int] | None = None,
) -> list[tuple[float, int, float, float, int]]:
    """
    Greedily select `num_samples` candidate queries using Optuna.

    Returns a list of tuples:
        (u_percent, IV, density, n_u_235, fuel_schedule)
    """
    write_path = Path(write_dir)
    write_path.mkdir(parents=True, exist_ok=True)

    if schedule_ids is None:
        schedule_ids = available_schedule_ids()
    schedule_ids = list(schedule_ids)

    best_queries: list[tuple[float, int, float, float, int]] = []

    for i in range(num_samples):
        study = optuna.create_study(direction="maximize")
        objective = make_objective(
            sequence_ensemble,
            prior_points=best_queries,
            gamma=gamma,
            schedule_ids=schedule_ids,
        )
        study.optimize(objective, n_trials=n_trials)

        study.trials_dataframe().to_csv(write_path / f"sample_{i}_trials.csv", index=False)

        params = study.best_trial.params
        u_percent = float(params["u_percent"])
        u_density = float(params["u_density"])
        density = float(params["density"])
        IV = int(params["IV"])
        fuel_schedule = int(params["fuel_schedule"])
        n_u_235 = compute_n_u235(
            u_percent,
            density,
            heavy_metal_fraction=u_density,
        )

        query = (u_percent, IV, density, n_u_235, fuel_schedule)
        best_queries.append(query)

        print(f"Selected query {i + 1}/{num_samples}: {query}")
        print(f"Best objective value: {study.best_value}")

    with open(write_path / "best_queries.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["u_percent", "IV", "density", "n_u_235", "fuel_schedule"])
        writer.writerows(best_queries)

    return best_queries


if __name__ == "__main__":
    # These should match the training dataset normalization values you used for
    # the saved ensemble. Keeping your original hard-coded values here for now.
    x_mean = torch.Tensor(
        [[
            0.5998, 0.3474, 2.5385, 2.0769, 2.0000,
            0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667,
            0.2849, 0.4783,
        ]]
    )

    x_std = torch.Tensor(
        [[
            0.1667, 0.4440, 0.8436, 0.7305, 0.8174,
            0.3731, 0.3731, 0.3731, 0.3731, 0.3731, 0.3731,
            0.3636, 0.3135,
        ]]
    )
    y_mean = torch.Tensor([171.1579])
    y_std = torch.Tensor([105.260])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_ensemble = SequenceEnsemble(
        "ensembles",
        x_mean.to(device),
        x_std.to(device),
        y_mean.to(device),
        y_std.to(device),
        device=device,
    )

    average_score = seq_ensemble.average_over_selection(
        0.8,
        1,
        13630,
        2.36e20,
        0,
    )
    print(f"average_score: {average_score}")
    print("now trying optuna")

    find_best_queries(
        sequence_ensemble=seq_ensemble,
        n_trials=100,
        num_samples=3,
        gamma=2,
        write_dir="outputs/test_run_3",
    )
