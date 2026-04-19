# utils.py
from __future__ import annotations

from typing import Dict, List, Tuple, Sequence, Optional, Callable
import numpy as np
import pandas as pd
import torch
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from datetime import datetime
from uuid import uuid4
from typing import Union
import torch
import torch.nn as nn

AVOGADRO = 6.02214076e23  # atoms / mol

# Replace these with your true fixed valid reactor positions if you know them.
# Otherwise you can auto-infer them from the historical dataset using
# infer_vehicle_static_positions_from_df(...)
VEHICLE_STATIC_POSITIONS: Dict[int, List[Tuple[int, int, int]]] = {
    1: [(1, 1, 1),
 (1, 1, 2),
 (1, 1, 3),
 (1, 1, 4),
 (1, 1, 5),
 (1, 1, 6),
 (1, 2, 1),
 (1, 2, 2),
 (1, 2, 3),
 (1, 2, 4),
 (1, 2, 5),
 (1, 2, 6),
 (1, 3, 1),
 (1, 3, 2),
 (1, 3, 3),
 (1, 3, 4),
 (1, 3, 5),
 (1, 3, 6),
 (2, 1, 1),
 (2, 1, 2),
 (2, 1, 3),
 (2, 1, 4),
 (2, 1, 5),
 (2, 1, 6),
 (2, 2, 1),
 (2, 2, 2),
 (2, 2, 3),
 (2, 2, 4),
 (2, 2, 5),
 (2, 2, 6),
 (2, 3, 1),
 (2, 3, 2),
 (2, 3, 3),
 (2, 3, 4),
 (2, 3, 5),
 (2, 3, 6),
 (3, 1, 1),
 (3, 1, 2),
 (3, 1, 3),
 (3, 1, 4),
 (3, 1, 5),
 (3, 1, 6),
 (3, 2, 1),
 (3, 2, 2),
 (3, 2, 3),
 (3, 2, 4),
 (3, 2, 5),
 (3, 2, 6),
 (3, 3, 1),
 (3, 3, 2),
 (3, 3, 3),
 (3, 3, 4),
 (3, 3, 5),
 (3, 3, 6),
 (4, 1, 1),
 (4, 1, 2),
 (4, 1, 3),
 (4, 1, 4),
 (4, 1, 5),
 (4, 1, 6),
 (4, 2, 1),
 (4, 2, 2),
 (4, 2, 3),
 (4, 2, 4),
 (4, 2, 5),
 (4, 2, 6),
 (4, 3, 1),
 (4, 3, 2),
 (4, 3, 3),
 (4, 3, 4),
 (4, 3, 5),
 (4, 3, 6),
 (5, 1, 1),
 (5, 1, 2),
 (5, 1, 3),
 (5, 1, 4),
 (5, 1, 5),
 (5, 1, 6),
 (5, 2, 1),
 (5, 2, 2),
 (5, 2, 3),
 (5, 2, 4),
 (5, 2, 5),
 (5, 2, 6),
 (5, 3, 1),
 (5, 3, 2),
 (5, 3, 3),
 (5, 3, 4),
 (5, 3, 5),
 (5, 3, 6)],
    2: [(1, 1, 1),
 (1, 1, 2),
 (1, 1, 3),
 (1, 1, 4),
 (1, 1, 5),
 (1, 1, 6),
 (1, 2, 1),
 (1, 2, 2),
 (1, 2, 3),
 (1, 2, 4),
 (1, 2, 5),
 (1, 2, 6),
 (1, 3, 1),
 (1, 3, 2),
 (1, 3, 3),
 (1, 3, 4),
 (1, 3, 5),
 (1, 3, 6),
 (2, 1, 1),
 (2, 1, 2),
 (2, 1, 3),
 (2, 1, 4),
 (2, 1, 5),
 (2, 1, 6),
 (2, 2, 1),
 (2, 2, 2),
 (2, 2, 3),
 (2, 2, 4),
 (2, 2, 5),
 (2, 2, 6),
 (2, 3, 1),
 (2, 3, 2),
 (2, 3, 3),
 (2, 3, 4),
 (2, 3, 5),
 (2, 3, 6),
 (3, 1, 1),
 (3, 1, 2),
 (3, 1, 3),
 (3, 1, 4),
 (3, 1, 5),
 (3, 1, 6),
 (3, 2, 1),
 (3, 2, 2),
 (3, 2, 3),
 (3, 2, 4),
 (3, 2, 5),
 (3, 2, 6),
 (3, 3, 1),
 (3, 3, 2),
 (3, 3, 3),
 (3, 3, 4),
 (3, 3, 5),
 (3, 3, 6)],
}


def infer_vehicle_static_positions_from_df(
    df: pd.DataFrame,
) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    Infer valid (Digit1, Digit2, Digit3) tuples per vehicle from the existing data.

    Assumes the first 8 columns are:
    U%, Density, Thermal_Conductivity, IV, Digit1, Digit2, Digit3, N_U_235
    """
    iv_col = df.columns[3]
    d1_col = df.columns[4]
    d2_col = df.columns[5]
    d3_col = df.columns[6]

    out: Dict[int, List[Tuple[int, int, int]]] = {}
    for iv in sorted(df[iv_col].astype(int).unique()):
        sub = df[df[iv_col].astype(int) == int(iv)]
        positions = sorted(
            {
                (int(row[d1_col]), int(row[d2_col]), int(row[d3_col]))
                for _, row in sub.iterrows()
            }
        )
        out[int(iv)] = positions
    return out


def compute_n_u235(
    U_percent: float,
    density: float,
    *,
    density_units: str = "kg/m3",
    heavy_metal_fraction: float = 1.0,
) -> float:
    """
    Approximate N_U_235 from enrichment and density.

    Assumptions:
    - U_percent is in percent units, e.g. 0.711 means 0.711%
    - density is heavy-metal density unless you change heavy_metal_fraction
    - returns atoms / cm^3

    If later you add material type explicitly, this is the right place to
    update the physics.
    """
    enrichment_frac = float(U_percent) / 100.0

    if density_units.lower() in {"kg/m3", "kg/m^3"}:
        density_g_per_cm3 = float(density) / 1000.0
    elif density_units.lower() in {"g/cm3", "g/cc"}:
        density_g_per_cm3 = float(density)
    else:
        raise ValueError(f"Unsupported density_units={density_units!r}")

    # Slightly better than using a fixed 238:
    avg_u_mass = (
        enrichment_frac * 235.0439299
        + (1.0 - enrichment_frac) * 238.05078826
    )

    n_u_total = density_g_per_cm3 * heavy_metal_fraction * AVOGADRO / avg_u_mass
    n_u235 = n_u_total * enrichment_frac
    return float(n_u235)


def encode_iv(iv: int) -> int:
    col3_map = {2: 1, 1: 3}
    if iv not in col3_map:
        raise ValueError(f"Unsupported IV={iv}. Expected 1 or 2.")
    return col3_map[iv]


def encode_digit1(iv: int, digit1: int) -> int:
    if iv == 1:
        mapping = {1: 3, 2: 3, 5: 2, 3: 2, 4: 1}
    elif iv == 2:
        mapping = {2: 2, 3: 2, 1: 1}
    else:
        raise ValueError(f"Unsupported IV={iv}. Expected 1 or 2.")

    if digit1 not in mapping:
        raise ValueError(f"Digit1={digit1} is not valid for IV={iv}.")
    return mapping[digit1]


def one_hot_digit3(
    digit3: int,
    categories: Sequence[int] = (1, 2, 3, 4, 5, 6),
) -> np.ndarray:
    vec = np.zeros(len(categories), dtype=np.float32)
    try:
        idx = list(categories).index(int(digit3))
    except ValueError as e:
        raise ValueError(f"Digit3={digit3} not found in categories={categories}") from e
    vec[idx] = 1.0
    return vec


def lookup_r_a(
    iv: int,
    digit1: int,
    digit2: int,
    digit3: int,
    ras_df: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Same logic as build_RAS_mapper, but for one candidate position.
    """
    if iv == 2:
        match_r = ras_df.loc[ras_df.iloc[:, 3] == digit1]
        if match_r.empty:
            raise ValueError(f"No R match for IV=2, Digit1={digit1}")
        r_val = float(match_r.iloc[0, 4])

        match_a = ras_df.loc[
            (ras_df.iloc[:, 0] == digit2) &
            (ras_df.iloc[:, 1] == digit3)
        ]
        if match_a.empty:
            raise ValueError(f"No A match for IV=2, Digit2={digit2}, Digit3={digit3}")
        a_val = float(match_a.iloc[0, 2])

    elif iv == 1:
        match_r = ras_df.loc[ras_df.iloc[:, 8] == digit1]
        if match_r.empty:
            raise ValueError(f"No R match for IV=1, Digit1={digit1}")
        r_val = float(match_r.iloc[0, 9])

        match_a = ras_df.loc[
            (ras_df.iloc[:, 5] == digit2) &
            (ras_df.iloc[:, 6] == digit3)
        ]
        if match_a.empty:
            raise ValueError(f"No A match for IV=1, Digit2={digit2}, Digit3={digit3}")
        a_val = float(match_a.iloc[0, 7])

    else:
        raise ValueError(f"Unsupported IV={iv}. Expected 1 or 2.")

    return r_val, a_val


def load_default_scalers(
    scaler_dir: str = ".",
):
    density_scaler = joblib.load(f"{scaler_dir}/col1_scaler.pkl")
    n_u235_scaler = joblib.load(f"{scaler_dir}/col7_scaler.pkl")
    r_scaler = joblib.load(f"{scaler_dir}/R_scaler.pkl")
    a_scaler = joblib.load(f"{scaler_dir}/A_scaler.pkl")
    return density_scaler, n_u235_scaler, r_scaler, a_scaler


def expand_time_config(
    t_config,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Make one time config broadcast across all selected positions.

    Expected:
    - 1D tensor/array/list of length L  -> repeated to [batch_size, L]
    - 2D tensor with shape [1, L]       -> repeated to [batch_size, L]
    - 2D tensor with shape [batch_size, L] -> used directly
    """
    t = torch.as_tensor(t_config, device=device, dtype=dtype)

    if t.ndim == 1:
        return t.unsqueeze(0).repeat(batch_size, 1)
    if t.ndim == 2:
        if t.shape[0] == 1:
            return t.repeat(batch_size, 1)
        if t.shape[0] == batch_size:
            return t
        raise ValueError(
            f"2D time config has batch={t.shape[0]}, expected 1 or {batch_size}"
        )

    raise ValueError(f"time config must be 1D or 2D, got shape={tuple(t.shape)}")


def build_candidate_feature_batch(
    U_percent: float,
    density: float,
    IV: int,
    *,
    x_mean,
    x_std,
    ras_df: pd.DataFrame,
    density_scaler,
    n_u235_scaler,
    r_scaler,
    a_scaler,
    vehicle_positions: Dict[int, List[Tuple[int, int, int]]],
    digit3_categories: Sequence[int] = (1, 2, 3, 4, 5, 6),
    n_u235: Optional[float] = None,
    n_u235_fn: Optional[Callable[[float, float], float]] = None,
    density_units: str = "kg/m3",
    heavy_metal_fraction: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """
    Builds the same 14-D model input the loader creates, then z-scores it.
    """
    IV = int(IV)

    if IV not in vehicle_positions or len(vehicle_positions[IV]) == 0:
        raise ValueError(
            f"No valid positions found for IV={IV}. "
            f"Pass vehicle_positions explicitly or populate VEHICLE_STATIC_POSITIONS."
        )

    if n_u235 is None:
        if n_u235_fn is None:
            n_u235 = compute_n_u235(
                U_percent,
                density,
                density_units=density_units,
                heavy_metal_fraction=heavy_metal_fraction,
            )
        else:
            n_u235 = float(n_u235_fn(U_percent, density))

    rows = []
    metadata = []

    for digit1, digit2, digit3 in vehicle_positions[IV]:
        r_val, a_val = lookup_r_a(IV, digit1, digit2, digit3, ras_df)

        row = [
            float(U_percent),
            float(density_scaler.transform(np.array([[density]], dtype=np.float64))[0, 0]),
            float(encode_iv(IV)),
            float(encode_digit1(IV, digit1)),
            float(digit2),
            float(n_u235_scaler.transform(np.array([[n_u235]], dtype=np.float64))[0, 0]),
            *one_hot_digit3(digit3, categories=digit3_categories).tolist(),
            float(r_scaler.transform(np.array([[r_val]], dtype=np.float64))[0, 0]),
            float(1.0 - a_scaler.transform(np.array([[abs(a_val)]], dtype=np.float64))[0, 0]),
        ]
        rows.append(row)
        metadata.append(
            {
                "IV": IV,
                "Digit1": int(digit1),
                "Digit2": int(digit2),
                "Digit3": int(digit3),
                "R": float(r_val),
                "A": float(a_val),
            }
        )

    x = torch.tensor(rows, dtype=dtype, device=device)

    x_mean_t = torch.as_tensor(x_mean, dtype=dtype, device=device)
    x_std_t = torch.as_tensor(x_std, dtype=dtype, device=device)

    if x_mean_t.ndim == 1:
        x_mean_t = x_mean_t.unsqueeze(0)
    if x_std_t.ndim == 1:
        x_std_t = x_std_t.unsqueeze(0)

    x = (x - x_mean_t) / x_std_t
    return x, metadata, float(n_u235)

def save_module_unique(
    module: nn.Module,
    save_dir: Union[str, Path],
    prefix: str = "model",
    save_state_dict: bool = True,
) -> Path:
    """
    Save a torch.nn.Module to a directory using a unique filename.

    Args:
        module: The PyTorch module to save.
        save_dir: Directory to save into.
        prefix: Prefix for the filename.
        save_state_dict: If True, saves module.state_dict().
                         If False, saves the full module object.

    Returns:
        Path to the saved file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid4().hex[:8]
    filename = f"{prefix}_{timestamp}_{unique_id}.pt"
    save_path = save_dir / filename

    obj_to_save = module.state_dict() if save_state_dict else module
    torch.save(obj_to_save, save_path)

    return save_path


from pathlib import Path
from typing import Union
import torch
import torch.nn as nn


def load_module(
    module: nn.Module,
    load_path: Union[str, Path],
    map_location: str | torch.device = "cpu",
    expects_state_dict: bool = True,
) -> nn.Module:
    """
    Load weights or a full module from disk.

    Args:
        module: An already-constructed module. Required if expects_state_dict=True.
        load_path: Path to the saved file.
        map_location: Device mapping for torch.load.
        expects_state_dict: If True, loads a state_dict into `module`.
                            If False, loads and returns the full saved module.

    Returns:
        The loaded module.
    """
    load_path = Path(load_path)


    if expects_state_dict:
        loaded_obj = torch.load(load_path, map_location=map_location, weights_only=True)
        module.load_state_dict(loaded_obj)
        module.eval()
        return module
    else:
        loaded_obj = torch.load(load_path, map_location=map_location, weights_only=False)

        if not isinstance(loaded_obj, nn.Module):
            raise TypeError("Loaded object is not a torch.nn.Module.")
        loaded_obj.eval()
        return loaded_obj