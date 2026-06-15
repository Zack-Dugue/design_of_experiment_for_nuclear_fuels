import csv
import glob
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data.dataset import Dataset

from time_schedules import (
    N_STATIC_COLUMNS,
    infer_registered_schedule_id,
    parse_time_columns,
    resolve_time_values,
)
from utils import VEHICLE_STATIC_POSITIONS


# Step 1: Read all CSV files
file_paths = glob.glob("C:\\Users\\dugue\\Downloads\\Gustavo Code\\Code\\fuel/*.csv")


EXPECTED_STATIC_COLUMNS_7 = [
    "U%",
    "Density",
    "Thermal_Conductivity",
    "IV",
    "Digit1",
    "Digit2",
    "Digit3",
]


# This code is now intentionally standardized around the current processed CSV
# format:
#   U%, Density, Thermal_Conductivity, IV, Digit1, Digit2, Digit3, time0, time1, ...
# i.e. exactly 7 static columns.
if N_STATIC_COLUMNS != 7:
    raise ValueError(
        f"This patched load_data.py expects N_STATIC_COLUMNS=7, but got {N_STATIC_COLUMNS}. "
        "Set N_STATIC_COLUMNS = 7 in time_schedules.py, or regenerate CSVs with the 8-column format."
    )


# -----------------------------------------------------------------------------
# In-memory synthetic CSV cache
# -----------------------------------------------------------------------------
# Old calling pattern elsewhere in the project can stay exactly the same:
#
#     create_synthetic_csv("test.csv", ...)
#     dataset = HGRDataset(["test.csv"], ...)
#
# The difference is that create_synthetic_csv no longer has to write a real CSV.
# It stores a DataFrame in this module-level cache, and load_data checks this
# cache before falling back to pd.read_csv(path).
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


def _onehot_col6(col6):
    """Fast fixed one-hot equivalent to OneHotEncoder(categories=[[1..6]])."""
    col6 = np.asarray(col6)
    out = np.zeros((col6.shape[0], 6), dtype=np.float32)
    valid = (col6 >= 1) & (col6 <= 6) & np.isfinite(col6)
    if np.any(valid):
        out[np.where(valid)[0], col6[valid].astype(np.int64) - 1] = 1.0
    return out


def truncate_row(row, total_length):
    if len(row) < total_length:
        return None
    return row[:total_length]


def _validate_csv_header(path, df):
    cols = list(df.columns)
    if len(cols) <= N_STATIC_COLUMNS:
        raise ValueError(
            f"{path} has {len(cols)} columns, but expected at least "
            f"{N_STATIC_COLUMNS + 1} columns: {N_STATIC_COLUMNS} static columns plus time-series targets."
        )

    # Do not require exact capitalization, but warn early if the first seven columns
    # are not the format this loader is now written for.
    normalized = [str(c).strip().lower() for c in cols[:N_STATIC_COLUMNS]]
    expected = [c.lower() for c in EXPECTED_STATIC_COLUMNS_7]
    if normalized != expected:
        raise ValueError(
            f"{path} does not have the expected first {N_STATIC_COLUMNS} static columns.\n"
            f"Expected: {EXPECTED_STATIC_COLUMNS_7}\n"
            f"Got:      {cols[:N_STATIC_COLUMNS]}\n"
            "This usually means you are mixing old/new preprocessing outputs. Delete the processed CSVs and regenerate them."
        )


def _read_csv_or_cached(path):
    key = _cache_key(path)
    if key in _SYNTHETIC_CSV_CACHE:
        # Return a copy so downstream truncation/renaming cannot mutate the cache.
        return _SYNTHETIC_CSV_CACHE[key].copy()
    return pd.read_csv(path)


def load_data(file_paths, target_length=None):
    """
    Load HGR/BURNUP CSV files and return a combined dataframe plus per-row time values.

    This patched version assumes the processed CSVs have exactly 7 static columns:
        U%, Density, Thermal_Conductivity, IV, Digit1, Digit2, Digit3
    and every column after that is a time/value column.

    If target_length is None, the loader truncates every file to the shortest
    available time-series length in the provided file list.

    Speed patch:
        Paths previously produced by create_synthetic_csv are read from an
        in-memory cache instead of disk. This preserves the old call sequence
        while avoiding the slow write/read loop for synthetic query datasets.
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

    # Static labels are fixed. Time labels are normalized to 0..target_length-1
    # in the combined dataframe because individual files may have different
    # actual time headers. The true time values are stored separately in
    # time_value_list and returned by the Dataset.
    column_labels = EXPECTED_STATIC_COLUMNS_7 + [str(c) for c in range(target_length)]

    truncated_data = []
    time_value_list = []
    schedule_id_list = []

    for path, df in zip(file_paths, all_data):
        if len(df.columns) < total_columns_to_keep:
            raise ValueError(
                f"{path} only has {len(df.columns) - N_STATIC_COLUMNS} time columns, "
                f"but target_length={target_length}."
            )

        file_time_values = parse_time_columns(df.columns[N_STATIC_COLUMNS : N_STATIC_COLUMNS + target_length])
        file_time_values = np.asarray(file_time_values, dtype=np.float32)
        schedule_id = infer_registered_schedule_id(file_time_values)

        # Old code iterated row-by-row and rebuilt a dataframe from a list.
        # Slicing once is much faster and produces the same combined format.
        truncated_arr = df.iloc[:, :total_columns_to_keep].to_numpy(copy=True)
        truncated_df = pd.DataFrame(truncated_arr, columns=column_labels)
        truncated_data.append(truncated_df)

        n_rows = truncated_arr.shape[0]
        time_value_list.extend([file_time_values] * n_rows)
        schedule_id_list.extend([schedule_id] * n_rows)

    upsampled_data_combined = pd.concat(truncated_data, ignore_index=True)
    return upsampled_data_combined, time_value_list, schedule_id_list


def build_RAS_mapper(size, upsampled_data_combined, path="RAS.csv"):
    """
    Vectorized replacement for the original iterrows/DataFrame.loc loop.
    Returns the same shape: (size, 2), columns [R, A].
    """
    ras_df = pd.read_csv(path)
    ras_mapped = np.full((size, 2), np.nan, dtype=np.float32)

    data = upsampled_data_combined.iloc[:, :N_STATIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    iv = data.iloc[:, 3].to_numpy()
    d1 = data.iloc[:, 4].to_numpy()
    d2 = data.iloc[:, 5].to_numpy()
    d3 = data.iloc[:, 6].to_numpy()

    # Build lookup dictionaries once instead of filtering ras_df for every row.
    ras = ras_df.apply(pd.to_numeric, errors="coerce")

    # IV == 2 mappings
    r_map_iv2 = dict(zip(ras.iloc[:, 3], ras.iloc[:, 4]))
    a_map_iv2 = dict(zip(zip(ras.iloc[:, 0], ras.iloc[:, 1]), ras.iloc[:, 2]))

    # IV == 1 mappings
    r_map_iv1 = dict(zip(ras.iloc[:, 8], ras.iloc[:, 9]))
    a_map_iv1 = dict(zip(zip(ras.iloc[:, 5], ras.iloc[:, 6]), ras.iloc[:, 7]))

    mask2 = iv == 2
    if np.any(mask2):
        idx = np.where(mask2)[0]
        ras_mapped[idx, 0] = np.array([r_map_iv2.get(v, np.nan) for v in d1[idx]], dtype=np.float32)
        ras_mapped[idx, 1] = np.array(
            [a_map_iv2.get((a, b), np.nan) for a, b in zip(d2[idx], d3[idx])],
            dtype=np.float32,
        )

    mask1 = iv == 1
    if np.any(mask1):
        idx = np.where(mask1)[0]
        ras_mapped[idx, 0] = np.array([r_map_iv1.get(v, np.nan) for v in d1[idx]], dtype=np.float32)
        ras_mapped[idx, 1] = np.array(
            [a_map_iv1.get((a, b), np.nan) for a, b in zip(d2[idx], d3[idx])],
            dtype=np.float32,
        )

    return ras_mapped


def encode(upsampled_data_combined):
    """
    Fast vectorized encoder.

    Output is compatible with the previous encode():
        X_full columns:
            col0, col1, col5, col3_encoded, col4_encoded, onehot(col6: 1..6)
        y:
            all target/time columns
    """
    data = pd.DataFrame(upsampled_data_combined)

    y = data.iloc[:, N_STATIC_COLUMNS:].to_numpy(dtype=np.float32, copy=True)

    static = data.iloc[:, :N_STATIC_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    arr = static.to_numpy(dtype=np.float32, copy=False)

    col0 = arr[:, 0]
    col1 = arr[:, 1]
    col3 = arr[:, 3]
    col4 = arr[:, 4]
    col5 = arr[:, 5]
    col6 = arr[:, 6]

    col3_encoded = np.zeros_like(col3, dtype=np.float32)
    col3_encoded[col3 == 2] = 1.0
    col3_encoded[col3 == 1] = 3.0

    col4_encoded = np.zeros_like(col4, dtype=np.float32)

    # Original encode_col4(row):
    # if col3 == 1: {1:3, 2:3, 5:2, 3:2, 4:1}
    mask = col3 == 1
    col4_encoded[mask & ((col4 == 1) | (col4 == 2))] = 3.0
    col4_encoded[mask & ((col4 == 5) | (col4 == 3))] = 2.0
    col4_encoded[mask & (col4 == 4)] = 1.0

    # if col3 == 2: {2:2, 3:2, 1:1}
    mask = col3 == 2
    col4_encoded[mask & ((col4 == 2) | (col4 == 3))] = 2.0
    col4_encoded[mask & (col4 == 1)] = 1.0

    nominal_encoded = _onehot_col6(col6)

    X_full = np.concatenate(
        [
            col0[:, None],
            col1[:, None],
            col5[:, None],
            col3_encoded[:, None],
            col4_encoded[:, None],
            nominal_encoded,
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    return X_full, y


def RAS_Encode(X, ras_mapped):
    # encode() currently emits:
    #   col0, col1, col5, col3_encoded, col4_encoded, feature1..feature6
    n_base = 5
    n_onehot = X.shape[1] - n_base
    column_names = [
        "col0",
        "col1",
        "col5",
        "col3_encoded",
        "col4_encoded",
    ] + [f"feature{i + 1}" for i in range(n_onehot)]

    X_full_df = pd.DataFrame(X, columns=column_names)
    X_full_df["R"] = ras_mapped[:, 0]
    X_full_df["A"] = ras_mapped[:, 1]

    # Aggregate/unknown RAS rows may not map. Keep the code runnable by using 0
    # for missing mapped coordinates rather than letting NaNs poison training.
    X_full_df["R"] = pd.to_numeric(X_full_df["R"], errors="coerce").fillna(0.0)
    X_full_df["A"] = pd.to_numeric(X_full_df["A"], errors="coerce").fillna(0.0)

    scalers = {}
    for col in ["col1", "R", "A"]:
        if col == "A":
            abs_A = np.abs(X_full_df[col].values.reshape(-1, 1))
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_abs = scaler.fit_transform(abs_A)
            X_full_df[col + "_scaled"] = 1 - scaled_abs
        else:
            scaler = MinMaxScaler()
            X_full_df[col + "_scaled"] = scaler.fit_transform(X_full_df[[col]])
        scalers[col] = scaler
        joblib.dump(scaler, f"{col}_scaler.pkl")

    selected_cols = [
        "col0",
        "col1_scaled",
        "col3_encoded",
        "col4_encoded",
        "col5",
    ] + [f"feature{i + 1}" for i in range(n_onehot)] + [
        "R_scaled",
        "A_scaled",
    ]

    X_df = pd.DataFrame(X_full_df[selected_cols])
    return X_df


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
    ):
        super().__init__()
        upsampled_data_combined, time_value_list, schedule_id_list = load_data(
            file_paths,
            target_length=target_length,
        )

        # Leave these off by default because they become noisy and slow when
        # synthetic query datasets are created repeatedly. Set this env var if
        # you need the old debugging output:
        #   set HGR_DATASET_DEBUG=1        # Windows cmd
        #   $env:HGR_DATASET_DEBUG="1"     # PowerShell
        #   export HGR_DATASET_DEBUG=1     # bash
        if os.environ.get("HGR_DATASET_DEBUG", "0") == "1":
            print("\nDEBUG AFTER load_data")
            print("combined shape:", upsampled_data_combined.shape)
            print("N_STATIC_COLUMNS:", N_STATIC_COLUMNS)
            print("len(time_value_list[0]):", len(time_value_list[0]))
            print("first 12 combined columns:", list(upsampled_data_combined.columns[:12]))
            print("last 10 combined columns:", list(upsampled_data_combined.columns[-10:]))

        ras_mapper = build_RAS_mapper(len(upsampled_data_combined), upsampled_data_combined)
        X, y = encode(upsampled_data_combined)
        X = RAS_Encode(X, ras_mapper)

        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).float()
        y = torch.from_numpy(np.asarray(y, dtype=np.float32)).float()
        time_values = torch.from_numpy(np.stack(time_value_list, axis=0)).float()

        if time_values.shape[1] != y.shape[1]:
            raise ValueError(
                f"Internal dataset mismatch: time_values has length {time_values.shape[1]}, "
                f"but y has length {y.shape[1]}. Check N_STATIC_COLUMNS and CSV headers."
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
        print(f"In HGRDataset X device = {X.device} , x_mean device = {x_mean.device}")
        X = (X - x_mean) / x_std
        y = (y - y_mean) / y_std

        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.X = X
        self.y = y
        self.time_values = time_values
        self.schedule_id_list = schedule_id_list
        self.return_time_values = return_time_values
        self.forced_spacing = None

    def over_ride_spacing(self, t):
        """
        Backward-compatible name from the original code.

        t can now be either:
            - an integer schedule id; or
            - an explicit time vector.
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


def create_synthetic_csv(path, U_percent, IV, density, n_u_235, t, MAX_LEN=120):
    """
    Backward-compatible synthetic query creator.

    IMPORTANT SPEED CHANGE:
        This function no longer writes a physical CSV by default. Instead, it
        creates the exact same synthetic table in memory and stores it in
        _SYNTHETIC_CSV_CACHE under the requested path. Then existing code like

            create_synthetic_csv("test.csv", ...)
            dataset = HGRDataset(["test.csv"], ...)

        still works, because load_data checks the cache before using pd.read_csv.

    The n_u_235 argument is kept for backward-compatible call sites, but is not
    written as a static column because the current processed CSV format has
    exactly 7 static columns.
    """
    time_series = _resolve_time_values_np(t, MAX_LEN)

    positions = np.asarray(VEHICLE_STATIC_POSITIONS[IV], dtype=np.float32)
    n_rows = positions.shape[0]

    static_block = np.column_stack(
        [
            np.full(n_rows, U_percent, dtype=np.float32),
            np.full(n_rows, density, dtype=np.float32),
            np.zeros(n_rows, dtype=np.float32),
            np.full(n_rows, IV, dtype=np.float32),
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
        ]
    )

    y_block = np.zeros((n_rows, len(time_series)), dtype=np.float32)
    arr = np.concatenate([static_block, y_block], axis=1)

    # Keep the actual time values in the headers, matching the previous on-disk
    # CSV behavior. load_data will parse these headers into time_value_list.
    header = EXPECTED_STATIC_COLUMNS_7 + [str(float(v)) for v in time_series.tolist()]
    df = pd.DataFrame(arr, columns=header)

    _SYNTHETIC_CSV_CACHE[_cache_key(path)] = df

    # Avoid accidentally reading an old stale test.csv if another path variant
    # misses the cache later. This is best-effort only.
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

    return path


if __name__ == "__main__":
    import time

    start = time.time()
    file_paths = glob.glob("C:\\Users\\dugue\\Downloads\\Gustavo Code\\Code\\fuel/*.csv")
    example_dataset = HGRDataset(file_paths, x_mean=None, x_std=None, y_mean=None, y_std=None)
    end = time.time()
    print("time to make the dataset: ", end - start)

    create_synthetic_csv("test.csv", 0.8, 2, 10920, 1, 0, MAX_LEN=72)
    dataset = HGRDataset(
        ["test.csv"],
        x_mean=example_dataset.x_mean,
        x_std=example_dataset.x_std,
        y_mean=example_dataset.y_mean,
        y_std=example_dataset.y_std,
    )
    X, t, y = dataset[32]
    print(f"X = {X}")
    print(f"t shape = {getattr(t, 'shape', None)}, t = {t}")
    print(f"y = {y}")
