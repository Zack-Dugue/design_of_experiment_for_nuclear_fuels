"""
Find high-exploration candidate queries for the nuclear-fuels HGR ensemble.

Compatible with the 9-static-column processed CSV format:
    Enrichment, TD_Density, Irradiation_Vehicle, R, A, S,
    N_U-235, Radial_R, Axial_Z, time0, time1, ...

Important tuple convention used throughout this file:
    query = (enrichment, IV, td_density, n_u_235, fuel_schedule)
which matches SequenceEnsemble.average_over_selection(...) and
SequenceEnsemble.compute_distance(...).

Notes:
    - This intentionally keeps global hard-coded normalization values.
      Replace X_MEAN_VALUES / X_STD_VALUES / Y_MEAN_VALUE / Y_STD_VALUE with
      the values from the ensemble's training dataset.
    - TD_Density in the new CSV is in g/cm^3-ish units, e.g. 9.864.
      The old compute_n_u235 calls appeared to use kg/m^3-ish values, e.g. 9864,
      so N_U235_DENSITY_SCALE defaults to 1000.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import csv

import optuna
import torch

from train_ensemble import SequenceEnsemble
from utils import compute_n_u235

try:
    from time_schedules import TIME_SCHEDULE_GAPS
except Exception:
    # Backward-compatible fallback for the original two-schedule repo.
    TIME_SCHEDULE_GAPS = {
        0: [0.001, 0.999, 4, 5, 5, 5, 5],
        1: [0.001, 0.999, 2, 2, 5, 5, 5, 2, 2, 2],
    }


# -----------------------------------------------------------------------------
# Global knobs
# -----------------------------------------------------------------------------

# The patched 9-column loader currently emits 12 model features:
#   Enrichment,
#   TD_Density,
#   Irradiation_Vehicle code,
#   R,
#   A,
#   log10(N_U-235),
#   onehot(S=1..6)
NUM_STATIC_FEATURES = 14

# Set this to the horizon you trained/evaluate the ensemble with. Your uploaded
# example CSV had 60 target columns, while older scripts often used 72.
EXPLORATION_TARGET_LENGTH = 60

# New CSV feature unit/range. Values like 9.864 appear in the processed CSV.
TD_DENSITY_RANGE = (8.0, 12.5)
ENRICHMENT_RANGE = (0.35, 5.0)
HEAVY_METAL_FRACTION_RANGE = (0.8, 1.0)
IV_RANGE = (1, 2)

# Old exploration code passed density values like 9000..16000 into compute_n_u235.
# If compute_n_u235 expects g/cm^3 instead of kg/m^3, change this to 1.0.
N_U235_DENSITY_SCALE = 1000.0


# -----------------------------------------------------------------------------
# Hard-coded normalization values
# -----------------------------------------------------------------------------
# Replace these with the real values from the training dataset. They are kept
# hard-coded by request, but the script now validates the length so a stale
# 13-feature vector fails immediately instead of corrupting scores.

X_MEAN_VALUES = [1.9656e+00, 1.0036e+01, 1.0909e+00, 2.1515e+00, 2.0000e+00, 1.8206e+01,
         1.6667e-01, 1.6667e-01, 1.6667e-01, 1.6667e-01, 1.6667e-01, 1.6667e-01,
         2.8390e+02, 1.3400e+01]
X_STD_VALUES = [  2.1502,   3.7289,   0.2876,   0.7437,   0.8167,   6.1182,   0.3728,
           0.3728,   0.3728,   0.3728,   0.3728,   0.3728,  35.8770, 168.7099]

Y_MEAN_VALUE = 265.8306
Y_STD_VALUE = 172.4701


def make_hardcoded_normalization() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(X_MEAN_VALUES) != NUM_STATIC_FEATURES:
        raise ValueError(
            f"X_MEAN_VALUES has {len(X_MEAN_VALUES)} values, but the 9-column loader emits "
            f"{NUM_STATIC_FEATURES} features. Replace it with the new 12-feature stats."
        )
    if len(X_STD_VALUES) != NUM_STATIC_FEATURES:
        raise ValueError(
            f"X_STD_VALUES has {len(X_STD_VALUES)} values, but the 9-column loader emits "
            f"{NUM_STATIC_FEATURES} features. Replace it with the new 12-feature stats."
        )

    x_mean = torch.tensor([X_MEAN_VALUES], dtype=torch.float32)
    x_std = torch.tensor([X_STD_VALUES], dtype=torch.float32)
    y_mean = torch.tensor([Y_MEAN_VALUE], dtype=torch.float32)
    y_std = torch.tensor([Y_STD_VALUE], dtype=torch.float32)

    x_std[x_std == 0] = 1.0
    y_std[y_std == 0] = 1.0
    return x_mean, x_std, y_mean, y_std


# -----------------------------------------------------------------------------
# Schedule helpers
# -----------------------------------------------------------------------------

def available_schedule_ids() -> list[int]:
    """Return sorted integer schedule IDs that Optuna is allowed to choose."""
    return sorted(int(k) for k in TIME_SCHEDULE_GAPS.keys())


def validate_schedule_ids(schedule_ids: Sequence[int] | None) -> list[int]:
    """Normalize and validate requested schedule IDs."""
    available = available_schedule_ids()
    if schedule_ids is None:
        return available

    requested = [int(s) for s in schedule_ids]
    bad = sorted(set(requested) - set(available))
    if bad:
        raise ValueError(
            f"Requested unknown schedule IDs: {bad}. Available schedule IDs: {available}. "
            "Add the missing schedule to TIME_SCHEDULE_GAPS or change schedule_ids."
        )
    if not requested:
        raise ValueError("schedule_ids was empty.")
    return requested


# -----------------------------------------------------------------------------
# Query helpers
# -----------------------------------------------------------------------------

Query = tuple[float, int, float, float, int]


def compute_query_n_u235(
    enrichment: float,
    td_density: float,
    heavy_metal_fraction: float,
) -> float:
    """
    Compute N_U-235 for a query.

    The synthetic CSV should receive td_density in the same units as TD_Density
    in the processed CSV. But compute_n_u235 historically appears to expect the
    old density scale, so we scale only for this computation.
    """
    return float(
        compute_n_u235(
            enrichment,
            td_density * N_U235_DENSITY_SCALE,
            heavy_metal_fraction=heavy_metal_fraction,
        )
    )


def tensor_to_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def infer_model_num_features(sequence_ensemble: SequenceEnsemble) -> int | None:
    """Try to infer the model input feature count from the loaded ensemble."""
    if not getattr(sequence_ensemble, "ensemble_list", None):
        return None
    model = sequence_ensemble.ensemble_list[0]
    try:
        return int(model.feature_encoder.linear.in_features)
    except Exception:
        return None


def validate_normalization_against_ensemble(sequence_ensemble: SequenceEnsemble) -> None:
    """Fail fast if hard-coded normalization does not match the saved model."""
    model_num_features = infer_model_num_features(sequence_ensemble)
    if model_num_features is None:
        print("Warning: could not infer model input feature count from ensemble.")
        return

    x_mean_features = int(sequence_ensemble.x_mean.shape[-1])
    x_std_features = int(sequence_ensemble.x_std.shape[-1])

    if x_mean_features != model_num_features or x_std_features != model_num_features:
        raise ValueError(
            "Hard-coded normalization shape does not match the saved ensemble.\n"
            f"  model expects: {model_num_features} features\n"
            f"  x_mean has:    {x_mean_features} features\n"
            f"  x_std has:     {x_std_features} features\n"
            "Update X_MEAN_VALUES and X_STD_VALUES to the new 12-feature stats."
        )


# -----------------------------------------------------------------------------
# Optuna objective and greedy selection
# -----------------------------------------------------------------------------

def make_objective(
    sequence_ensemble: SequenceEnsemble,
    prior_points: Sequence[Query],
    gamma: float = 0.0,
    schedule_ids: Sequence[int] | None = None,
    target_length: int = EXPLORATION_TARGET_LENGTH,
):
    """
    Build an Optuna objective for selecting high-uncertainty candidate queries.

    The base score is ensemble predictive variance from
    SequenceEnsemble.average_over_selection(...).

    If prior_points is non-empty and gamma > 0, add a diversity term based on
    the minimum model-output distance to already-selected points.
    """
    schedule_ids = validate_schedule_ids(schedule_ids)

    def objective(trial: optuna.Trial) -> float:
        enrichment = trial.suggest_float("enrichment", *ENRICHMENT_RANGE)
        heavy_metal_fraction = trial.suggest_float("heavy_metal_fraction", *HEAVY_METAL_FRACTION_RANGE)
        td_density = trial.suggest_float("td_density", *TD_DENSITY_RANGE)
        IV = trial.suggest_int("IV", *IV_RANGE)
        fuel_schedule = int(trial.suggest_categorical("fuel_schedule", schedule_ids))

        n_u_235 = compute_query_n_u235(
            enrichment=enrichment,
            td_density=td_density,
            heavy_metal_fraction=heavy_metal_fraction,
        )

        uncertainty_score = sequence_ensemble.average_over_selection(
            enrichment,
            IV,
            td_density,
            n_u_235,
            fuel_schedule,
            target_length=target_length,
        )
        uncertainty_score = tensor_to_float(uncertainty_score)

        diversity_score = 0.0
        candidate: Query = (enrichment, IV, td_density, n_u_235, fuel_schedule)

        if prior_points and gamma != 0:
            distances = []
            for prior_point in prior_points:
                d = sequence_ensemble.compute_distance(
                    candidate,
                    prior_point,
                    target_length=target_length,
                )
                distances.append(tensor_to_float(d))

            # Minimum distance encourages separation from the closest previously
            # selected query.
            diversity_score = min(distances)
            print(f"Diversity score: {diversity_score}")

        return uncertainty_score + gamma * diversity_score

    return objective


def find_best_queries(
    sequence_ensemble: SequenceEnsemble,
    write_dir: str | Path = "out",
    n_trials: int = 100,
    num_samples: int = 3,
    gamma: float = 0.0,
    schedule_ids: Sequence[int] | None = None,
    target_length: int = EXPLORATION_TARGET_LENGTH,
) -> list[Query]:
    """
    Greedily select `num_samples` candidate queries using Optuna.

    Returns a list of tuples:
        (enrichment, IV, td_density, n_u_235, fuel_schedule)
    """
    write_path = Path(write_dir)
    write_path.mkdir(parents=True, exist_ok=True)

    schedule_ids = validate_schedule_ids(schedule_ids)
    best_queries: list[Query] = []

    for i in range(num_samples):
        study = optuna.create_study(direction="maximize")
        objective = make_objective(
            sequence_ensemble,
            prior_points=best_queries,
            gamma=gamma,
            schedule_ids=schedule_ids,
            target_length=target_length,
        )
        study.optimize(objective, n_trials=n_trials)

        study.trials_dataframe().to_csv(write_path / f"sample_{i}_trials.csv", index=False)

        params = study.best_trial.params
        enrichment = float(params["enrichment"])
        heavy_metal_fraction = float(params["heavy_metal_fraction"])
        td_density = float(params["td_density"])
        IV = int(params["IV"])
        fuel_schedule = int(params["fuel_schedule"])

        n_u_235 = compute_query_n_u235(
            enrichment=enrichment,
            td_density=td_density,
            heavy_metal_fraction=heavy_metal_fraction,
        )

        query: Query = (enrichment, IV, td_density, n_u_235, fuel_schedule)
        best_queries.append(query)

        print(f"Selected query {i + 1}/{num_samples}: {query}")
        print(f"Best objective value: {study.best_value}")

    with open(write_path / "best_queries.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "enrichment",
                "IV",
                "td_density",
                "n_u_235",
                "fuel_schedule",
            ]
        )
        writer.writerows(best_queries)

    return best_queries


# -----------------------------------------------------------------------------
# Main smoke run
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    x_mean, x_std, y_mean, y_std = make_hardcoded_normalization()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    seq_ensemble = SequenceEnsemble(
        "ensembles",
        x_mean,
        x_std,
        y_mean,
        y_std,
        device=device,
    )
    validate_normalization_against_ensemble(seq_ensemble)

    # Smoke test candidate. TD density is now on the new CSV scale, e.g. 9.864.
    smoke_enrichment = 0.8
    smoke_IV = 1
    smoke_td_density = 9.864
    smoke_heavy_metal_fraction = 0.90
    smoke_schedule = 0
    smoke_n_u235 = compute_query_n_u235(
        enrichment=smoke_enrichment,
        td_density=smoke_td_density,
        heavy_metal_fraction=smoke_heavy_metal_fraction,
    )

    average_score = seq_ensemble.average_over_selection(
        smoke_enrichment,
        smoke_IV,
        smoke_td_density,
        smoke_n_u235,
        smoke_schedule,
        target_length=EXPLORATION_TARGET_LENGTH,
    )
    print(f"average_score: {average_score}")
    print("now trying optuna")

    find_best_queries(
        sequence_ensemble=seq_ensemble,
        n_trials=100,
        num_samples=3,
        gamma=2,
        write_dir="outputs/test_run_9col",
        schedule_ids=[2],
        target_length=EXPLORATION_TARGET_LENGTH,
    )
