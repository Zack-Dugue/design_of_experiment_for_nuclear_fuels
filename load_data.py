import glob
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

from time_schedules import (
    infer_registered_schedule_id,
    parse_time_columns,
    resolve_time_values,
)
from utils import VEHICLE_STATIC_POSITIONS


# =============================================================================
# Path settings
# =============================================================================

# Pick exactly one target family at a time. Do not mix Total_HGR and BURNUP CSVs
# in one training dataset.
PROJECT_ROOT = r"C:\Users\dugue\PycharmProjects\design_of_experiment_for_nuclear_fuels"

# Use this if you copied/wrote the wide Fuel Total_HGR CSVs into project/HGR_fuel:
DEFAULT_CSV_GLOB = os.path.join(PROJECT_ROOT, "HGR_fuel", "*.csv")

# Or use this if you want to load straight from the wide preprocessor output:
# DEFAULT_CSV_GLOB = os.path.join(
#     PROJECT_ROOT,
#     "RawFuels",
#     "processed_pinns",
#     "Fuel",
#     "Total_HGR",
#     "*.csv",
# )

# For Burnup, use one of these instead:
# DEFAULT_CSV_GLOB = os.path.join(PROJECT_ROOT, "BURNUP_fuel", "*.csv")
# DEFAULT_CSV_GLOB = os.path.join(
#     PROJECT_ROOT,
#     "RawFuels",
#     "processed_pinns",
#     "Fuel",
#     "BURNUP",
#     "*.csv",
# )

file_paths = [p for p in glob.glob(DEFAULT_CSV_GLOB) if os.path.isfile(p)]


# =============================================================================
# New wide CSV schema
# =============================================================================

EXPECTED_STATIC_COLUMNS_9 = [
    "Enrichment",
    "TD_Density",
    "Irradiation_Vehicle",
    "R",
    "A",
    "S",
    "N_U-235",
    "Radial_R",
    "Axial_Z",
]

# This loader owns its CSV schema. Do not import this from time_schedules.py.
# time_schedules.py should only be responsible for interpreting the time headers.
N_STATIC_COLUMNS = len(EXPECTED_STATIC_COLUMNS_9)


# =============================================================================
# In-memory synthetic CSV cache
# =============================================================================

_SYNTHETIC_CSV_CACHE: Dict[str, pd.DataFrame] = {}


def _cache_key(path):
    return os.path.abspath(os.fspath(path))


def _resolve_time_values_np(t, max_len):
    time_values = resolve_time_values(t, max_len=max_len)

    if isinstance(time_values, torch.Tensor):
        time_values = time_values.detach().cpu().numpy()
    else:
        time_values = np.asarray(time_values)

    return time_values[:max_len].astype(np.float32, copy=False)


def _onehot_int(values, min_val, max_val):
    values = np.asarray(values)
    out = np.zeros((values.shape[0], max_val - min_val + 1), dtype=np.float32)

    valid = np.isfinite(values) & (values >= min_val) & (values <= max_val)

    if np.any(valid):
        out[np.where(valid)[0], values[valid].astype(np.int64) - min_val] = 1.0

    return out


def truncate_row(row, total_length):
    if len(row) < total_length:
        return None
    return row[:total_length]


# =============================================================================
# CSV reading / validation
# =============================================================================


def _normalize_header_name(col):
    return str(col).strip().lower()


def _validate_csv_header(path, df):
    cols = list(df.columns)

    if "timestepms" in cols:
        raise ValueError(
            f"{path} is still in the OLD LONG format. It has a 'timestepms' column.\n"
            "Delete the old processed CSVs and rerun the fixed wide preprocessor.\n"
            "The loader now expects one row per RAS position and one column per timestep."
        )

    if len(cols) <= N_STATIC_COLUMNS:
        raise ValueError(
            f"{path} has {len(cols)} columns, but expected at least "
            f"{N_STATIC_COLUMNS + 1} columns: {N_STATIC_COLUMNS} static columns plus time-series targets."
        )

    normalized = [_normalize_header_name(c) for c in cols[:N_STATIC_COLUMNS]]
    expected = [c.lower() for c in EXPECTED_STATIC_COLUMNS_9]

    if normalized != expected:
        raise ValueError(
            f"{path} does not have the expected first {N_STATIC_COLUMNS} static columns.\n"
            f"Expected: {EXPECTED_STATIC_COLUMNS_9}\n"
            f"Got:      {cols[:N_STATIC_COLUMNS]}\n"
            "This usually means you are mixing old/new preprocessing outputs. "
            "Delete the processed CSVs and regenerate them with preprocess_pinns_wide.py."
        )

    # Check that the remaining columns are parseable as time values.
    raw_time_cols = cols[N_STATIC_COLUMNS:]
    try:
        parse_time_columns(raw_time_cols)
    except Exception as e:
        raise ValueError(
            f"{path} has invalid timestep columns after the first {N_STATIC_COLUMNS} static columns.\n"
            f"Time columns were: {raw_time_cols[:10]} ...\n"
            f"Original error: {e}"
        )


def _read_csv_or_cached(path):
    key = _cache_key(path)

    if key in _SYNTHETIC_CSV_CACHE:
        return _SYNTHETIC_CSV_CACHE[key].copy()

    return pd.read_csv(path)


def load_data(file_paths, target_length=None):
    """
    Load new wide-format HGR/BURNUP CSV files and return:

        upsampled_data_combined, time_value_list, schedule_id_list

    Expected CSV format:

        Enrichment, TD_Density, Irradiation_Vehicle, R, A, S,
        N_U-235, Radial_R, Axial_Z, 0, 1, 3, 5, ...

    i.e. exactly 9 static columns followed by timestep target columns.

    If target_length is None, every file is truncated to the shortest available
    time-series length in the provided file list. This preserves compatibility
    with folds that contain files with 40/50/60 timesteps.
    """
    if len(file_paths) == 0:
        raise ValueError("load_data received no file paths.")

    all_data = [_read_csv_or_cached(file) for file in file_paths]

    for path, df in zip(file_paths, all_data):
        _validate_csv_header(path, df)

    available_target_lengths = [len(df.columns) - N_STATIC_COLUMNS for df in all_data]

    if target_length is None:
        target_length = min(available_target_lengths)

    target_length = int(target_length)

    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")

    total_columns_to_keep = N_STATIC_COLUMNS + target_length

    column_labels = EXPECTED_STATIC_COLUMNS_9 + [str(c) for c in range(target_length)]

    truncated_data = []
    time_value_list = []
    schedule_id_list = []

    for path, df in zip(file_paths, all_data):
        available_len = len(df.columns) - N_STATIC_COLUMNS

        if available_len < target_length:
            raise ValueError(
                f"{path} only has {available_len} time columns, but target_length={target_length}."
            )

        file_time_values = parse_time_columns(
            df.columns[N_STATIC_COLUMNS : N_STATIC_COLUMNS + target_length]
        )
        file_time_values = np.asarray(file_time_values, dtype=np.float32)
        schedule_id = infer_registered_schedule_id(file_time_values)

        truncated_arr = df.iloc[:, :total_columns_to_keep].to_numpy(copy=True)
        truncated_df = pd.DataFrame(truncated_arr, columns=column_labels)
        truncated_data.append(truncated_df)

        n_rows = truncated_arr.shape[0]
        time_value_list.extend([file_time_values] * n_rows)
        schedule_id_list.extend([schedule_id] * n_rows)

    upsampled_data_combined = pd.concat(truncated_data, ignore_index=True)
    return upsampled_data_combined, time_value_list, schedule_id_list


# =============================================================================
# Encoding
# =============================================================================


def _vehicle_code(series):
    """
    Stable numeric encoding for Irradiation_Vehicle.

    New real CSVs use string labels:
        RB
        VXF

    Older synthetic calls may still pass numeric ids.

    Encoding:
        RB  -> 1
        VXF -> 2
    """
    series = pd.Series(series)

    s = series.astype(str).str.strip().str.upper()
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)

    mapping = {
        "RB": 1.0,
        "RABBIT": 1.0,
        "HFIR": 1.0,
        "VXF": 2.0,
        "VX-F": 2.0,
    }

    mapped = s.map(mapping).astype("float32").to_numpy()
    out = np.where(np.isfinite(numeric), numeric, mapped)

    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def _r_encoded(r_values, iv_code):
    """
    Preserves the old rough radial/category encoding behavior where possible.

    For RB/vehicle code 1:
        R 1/2 -> 3, R 5/3 -> 2, R 4 -> 1
    For VXF/vehicle code 2:
        R 2/3 -> 2, R 1 -> 1

    Unknown combinations become 0.
    """
    r_values = np.asarray(r_values, dtype=np.float32)
    iv_code = np.asarray(iv_code, dtype=np.float32)

    out = np.zeros_like(r_values, dtype=np.float32)

    rb = iv_code == 1
    out[rb & ((r_values == 1) | (r_values == 2))] = 3.0
    out[rb & ((r_values == 5) | (r_values == 3))] = 2.0
    out[rb & (r_values == 4)] = 1.0

    vxf = iv_code == 2
    out[vxf & ((r_values == 2) | (r_values == 3))] = 2.0
    out[vxf & (r_values == 1)] = 1.0

    return out


def encode(upsampled_data_combined):
    """
    Encode the new 9-static-column processed CSV format.

    Input static columns:
        Enrichment, TD_Density, Irradiation_Vehicle, R, A, S,
        N_U-235, Radial_R, Axial_Z

    Output feature columns:
        Enrichment
        TD_Density
        Irradiation_Vehicle code
        R_encoded
        A index
        log10(N_U-235)
        onehot(S: 1..6)
        Radial_R
        Axial_Z

    Feature count: 14.
    """
    data = pd.DataFrame(upsampled_data_combined)

    y = data.iloc[:, N_STATIC_COLUMNS:].to_numpy(dtype=np.float32, copy=True)

    enrichment = pd.to_numeric(data["Enrichment"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    density = pd.to_numeric(data["TD_Density"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    iv_code = _vehicle_code(data["Irradiation_Vehicle"])

    r = pd.to_numeric(data["R"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    a = pd.to_numeric(data["A"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    s = pd.to_numeric(data["S"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    n_u235 = pd.to_numeric(data["N_U-235"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    n_u235_log = np.log10(np.maximum(n_u235, 1.0)).astype(np.float32)

    radial_r = pd.to_numeric(data["Radial_R"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    axial_z = pd.to_numeric(data["Axial_Z"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    r_enc = _r_encoded(r, iv_code)
    s_onehot = _onehot_int(s, 1, 6)

    X_full = np.concatenate(
        [
            enrichment[:, None],
            density[:, None],
            iv_code[:, None],
            r_enc[:, None],
            a[:, None],
            n_u235_log[:, None],
            s_onehot,
            radial_r[:, None],
            axial_z[:, None],
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    return X_full, y


# =============================================================================
# Backward-compatible RAS helpers
# =============================================================================


def build_RAS_mapper(size, upsampled_data_combined, path="RAS.csv"):
    """
    New wide CSVs already contain physical coordinates.
    Return [Radial_R, Axial_Z] directly.
    """
    data = pd.DataFrame(upsampled_data_combined)

    if "Radial_R" not in data.columns or "Axial_Z" not in data.columns:
        raise ValueError(
            "build_RAS_mapper expected Radial_R and Axial_Z in the new CSV format. "
            "Regenerate CSVs with the fixed wide preprocessor."
        )

    out = np.column_stack(
        [
            pd.to_numeric(data["Radial_R"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(data["Axial_Z"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        ]
    )

    if out.shape[0] != size:
        raise ValueError(
            f"build_RAS_mapper size mismatch: requested size={size}, but data has {out.shape[0]} rows."
        )

    return out


def RAS_Encode(X, ras_mapped):
    """
    Kept only for old call sites.

    The new encode() already includes Radial_R and Axial_Z, so this function is
    now an identity operation. Do not call it in new code.
    """
    return pd.DataFrame(np.asarray(X, dtype=np.float32))


# =============================================================================
# Dataset
# =============================================================================


class HGRDataset(Dataset):
    def __init__(
        self,
        file_paths,
        x_mean=None,
        x_std=None,
        y_mean=None,
        y_std=None,
        target_length=None,
        return_time_values=True,
        device=None,
    ):
        super().__init__()

        upsampled_data_combined, time_value_list, schedule_id_list = load_data(
            file_paths,
            target_length=target_length,
        )

        if os.environ.get("HGR_DATASET_DEBUG", "0") == "1":
            print("\nDEBUG AFTER load_data")
            print("combined shape:", upsampled_data_combined.shape)
            print("N_STATIC_COLUMNS:", N_STATIC_COLUMNS)
            print("len(time_value_list[0]):", len(time_value_list[0]))
            print("first 12 combined columns:", list(upsampled_data_combined.columns[:12]))
            print("last 10 combined columns:", list(upsampled_data_combined.columns[-10:]))

        X, y = encode(upsampled_data_combined)

        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).float()
        y = torch.from_numpy(np.asarray(y, dtype=np.float32)).float()
        time_values = torch.from_numpy(np.stack(time_value_list, axis=0)).float()

        if time_values.shape[1] != y.shape[1]:
            raise ValueError(
                f"Internal dataset mismatch: time_values has length {time_values.shape[1]}, "
                f"but y has length {y.shape[1]}. Check CSV headers."
            )

        if x_mean is None:
            x_mean = X.mean(0).unsqueeze(0)

        if x_std is None:
            x_std = X.std(0).unsqueeze(0)

        x_std[x_std == 0] = 1

        if y_mean is None:
            y_mean = y.mean().unsqueeze(0)

        if y_std is None:
            y_std = y.std().unsqueeze(0)

        y_std[y_std == 0] = 1

        X = (X - x_mean) / x_std
        y = (y - y_mean) / y_std

        if device is None:
            self.x_mean = x_mean
            self.x_std = x_std
            self.y_mean = y_mean
            self.y_std = y_std
            self.X = X
            self.y = y
            self.time_values = time_values
        else:
            self.x_mean = x_mean.to(device)
            self.x_std = x_std.to(device)
            self.y_mean = y_mean.to(device)
            self.y_std = y_std.to(device)
            self.X = X.to(device)
            self.y = y.to(device)
            self.time_values = time_values.to(device)

        self.schedule_id_list = schedule_id_list
        self.return_time_values = return_time_values
        self.forced_spacing = None

    def over_ride_spacing(self, t):
        """
        Backward-compatible name from the original code.

        t can be either:
            - an integer schedule id
            - an explicit time vector
        """
        self.forced_spacing = t

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        if self.forced_spacing is not None:
            if isinstance(self.forced_spacing, (int, np.integer)):
                t = torch.tensor(int(self.forced_spacing), dtype=torch.long)
            else:
                t = torch.as_tensor(self.forced_spacing, dtype=torch.float32)

        elif self.return_time_values:
            t = self.time_values[idx]

        else:
            schedule_id = self.schedule_id_list[idx]

            if schedule_id is None:
                raise ValueError(
                    "This sample's time columns do not match a registered schedule. "
                    "Use return_time_values=True, or add this schedule to TIME_SCHEDULE_GAPS."
                )

            t = torch.tensor(int(schedule_id), dtype=torch.long)

        return X, t, y


# =============================================================================
# Synthetic query CSV compatibility
# =============================================================================


AXIAL_MAP = {
    "VXF": {
        (3, 6): 254.611, (3, 5): 230.111, (3, 4): 205.611,
        (3, 3): 181.111, (3, 2): 156.611, (3, 1): 132.111,
        (2, 6): 54.65, (2, 5): 30.15, (2, 4): 5.65,
        (2, 3): -18.85, (2, 2): -43.35, (2, 1): -67.85,
        (1, 6): -145.3115, (1, 5): -169.8115, (1, 4): -194.3115,
        (1, 3): -218.8115, (1, 2): -243.3115, (1, 1): -267.8115,
    },
    "RB": {
        (3, 6): 276.611, (3, 5): 252.111, (3, 4): 227.611,
        (3, 3): 203.111, (3, 2): 178.611, (3, 1): 154.111,
        (2, 6): 76.65, (2, 5): 52.15, (2, 4): 27.65,
        (2, 3): 3.15, (2, 2): -21.35, (2, 1): -45.85,
        (1, 6): -123.3115, (1, 5): -147.8115, (1, 4): -172.3115,
        (1, 3): -196.8115, (1, 2): -221.3115, (1, 1): -245.8115,
    },
}

RADIAL_MAP = {
    "VXF": {1: 401.32, 2: 387.505, 3: 387.505},
    "RB": {1: 260.3889, 2: 260.3889, 3: 277.9464, 4: 288.7, 5: 277.9464},
}


def _vehicle_label(iv):
    """
    Convert a caller's irradiation-vehicle value into the geometry-map label.

    Loader/model convention:
        RB  -> 1
        VXF -> 2

    Real CSVs usually contain string labels, while old exploration code often
    passes numeric ids.
    """
    if isinstance(iv, str):
        s = iv.strip().upper()
        if s in {"RB", "RABBIT", "HFIR", "1", "1.0"}:
            return "RB"
        if s in {"VXF", "VX-F", "2", "2.0"}:
            return "VXF"

    try:
        iv_code = int(float(iv))
    except (TypeError, ValueError):
        iv_code = int(float(_vehicle_code(pd.Series([iv]))[0]))

    if iv_code == 1:
        return "RB"
    if iv_code == 2:
        return "VXF"

    raise ValueError(f"Unknown Irradiation_Vehicle {iv!r}; expected RB/1 or VXF/2.")


def _physical_coords_from_ras(iv, r, a, s):
    """
    Deterministically reproduce the preprocessor's coordinate construction.

    Radial_R depends on (Irradiation_Vehicle, R).
    Axial_Z depends on (Irradiation_Vehicle, A, S).
    """
    iv_label = _vehicle_label(iv)
    r = int(r)
    a = int(a)
    s = int(s)

    try:
        radial_r = RADIAL_MAP[iv_label][r]
    except KeyError as exc:
        raise KeyError(
            f"No Radial_R mapping for IV={iv_label}, R={r}. "
            f"Known R values: {sorted(RADIAL_MAP[iv_label])}"
        ) from exc

    try:
        axial_z = AXIAL_MAP[iv_label][(a, s)]
    except KeyError as exc:
        raise KeyError(
            f"No Axial_Z mapping for IV={iv_label}, A={a}, S={s}. "
            f"Known (A,S) values: {sorted(AXIAL_MAP[iv_label])}"
        ) from exc

    return float(radial_r), float(axial_z)


def _vehicle_position_key(iv):
    if iv in VEHICLE_STATIC_POSITIONS:
        return iv

    iv_code = float(_vehicle_code(pd.Series([iv]))[0])

    candidates = []

    if np.isfinite(iv_code):
        candidates.extend([int(iv_code), float(iv_code), str(int(iv_code))])

    candidates.extend([str(iv).strip(), str(iv).strip().upper()])

    for candidate in candidates:
        if candidate in VEHICLE_STATIC_POSITIONS:
            return candidate

    raise KeyError(
        f"Could not find VEHICLE_STATIC_POSITIONS entry for Irradiation_Vehicle={iv!r}. "
        f"Tried candidates: {candidates}"
    )


def create_synthetic_csv(path, U_percent, IV, density, n_u_235, t, MAX_LEN=120):
    """
    Backward-compatible synthetic query creator for the new 9-column CSV schema.

    Creates a synthetic wide table in memory with columns:

        Enrichment, TD_Density, Irradiation_Vehicle, R, A, S,
        N_U-235, Radial_R, Axial_Z, timestep0, timestep1, ...

    Radial_R and Axial_Z are filled using the same deterministic geometry maps
    used by the preprocessing script:

        Radial_R = RADIAL_MAP[IV][R]
        Axial_Z  = AXIAL_MAP[IV][(A, S)]

    This keeps synthetic exploration queries on the same coordinate manifold as
    the real training CSVs.
    """
    time_series = _resolve_time_values_np(t, MAX_LEN)

    vehicle_key = _vehicle_position_key(IV)
    positions = np.asarray(VEHICLE_STATIC_POSITIONS[vehicle_key], dtype=np.float32)
    n_rows = positions.shape[0]

    if positions.shape[1] < 3:
        raise ValueError(
            f"VEHICLE_STATIC_POSITIONS[{vehicle_key!r}] must have at least three columns [R, A, S]. "
            f"Got shape {positions.shape}."
        )

    coords = np.array(
        [
            _physical_coords_from_ras(
                IV,
                positions[i, 0],  # R
                positions[i, 1],  # A
                positions[i, 2],  # S
            )
            for i in range(n_rows)
        ],
        dtype=np.float32,
    )

    radial_r = coords[:, 0]
    axial_z = coords[:, 1]

    static_block = np.column_stack(
        [
            np.full(n_rows, U_percent, dtype=np.float32),       # Enrichment
            np.full(n_rows, density, dtype=np.float32),         # TD_Density
            np.full(n_rows, IV, dtype=object),                  # Irradiation_Vehicle
            positions[:, 0].astype(np.float32),                 # R
            positions[:, 1].astype(np.float32),                 # A
            positions[:, 2].astype(np.float32),                 # S
            np.full(n_rows, n_u_235, dtype=np.float32),         # N_U-235
            radial_r,                                           # Radial_R
            axial_z,                                            # Axial_Z
        ]
    )

    y_block = np.zeros((n_rows, len(time_series)), dtype=np.float32)
    arr = np.concatenate([static_block, y_block], axis=1)

    header = EXPECTED_STATIC_COLUMNS_9 + [str(float(v)) for v in time_series.tolist()]
    df = pd.DataFrame(arr, columns=header)

    _SYNTHETIC_CSV_CACHE[_cache_key(path)] = df

    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

    return path


# =============================================================================
# Smoke test
# =============================================================================


if __name__ == "__main__":
    import time

    start = time.time()

    file_paths = [p for p in glob.glob(DEFAULT_CSV_GLOB) if os.path.isfile(p)]

    print(f"CSV glob    = {DEFAULT_CSV_GLOB}")
    print(f"CSV count   = {len(file_paths)}")

    if len(file_paths) == 0:
        raise ValueError(
            "No CSV files found. Check DEFAULT_CSV_GLOB and make sure it ends with *.csv."
        )

    example_dataset = HGRDataset(
        file_paths,
        x_mean=None,
        x_std=None,
        y_mean=None,
        y_std=None,
    )

    end = time.time()

    print("time to make the dataset:", end - start)
    print("X shape:", example_dataset.X.shape)
    print("y shape:", example_dataset.y.shape)
    print("time_values shape:", example_dataset.time_values.shape)

    create_synthetic_csv("test.csv", 0.8, 2, 9.864, 1, 0, MAX_LEN=72)
    dataset = HGRDataset(
        ["test.csv"],
        x_mean=example_dataset.x_mean,
        x_std=example_dataset.x_std,
        y_mean=example_dataset.y_mean,
        y_std=example_dataset.y_std,
    )

    X, t, y = dataset[0]
    print("Synthetic X shape:", X.shape)
    print("Synthetic t shape:", getattr(t, "shape", None))
    print("Synthetic y shape:", y.shape)
