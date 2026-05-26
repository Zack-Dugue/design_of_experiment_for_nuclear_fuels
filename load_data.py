import csv
import glob
import os

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


def load_data(file_paths, target_length=None):
    """
    Load HGR/BURNUP CSV files and return a combined dataframe plus per-row time values.

    This patched version assumes the processed CSVs have exactly 7 static columns:
        U%, Density, Thermal_Conductivity, IV, Digit1, Digit2, Digit3
    and every column after that is a time/value column.

    If target_length is None, the loader truncates every file to the shortest
    available time-series length in the provided file list.
    """
    if len(file_paths) == 0:
        raise ValueError("load_data received no file paths.")

    all_data = [pd.read_csv(file) for file in file_paths]
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
        schedule_id = infer_registered_schedule_id(file_time_values)

        truncated_rows = []
        for row in df[df.columns].values:
            truncated = truncate_row(row, total_columns_to_keep)
            if truncated is not None:
                truncated_rows.append(truncated)
                time_value_list.append(file_time_values.copy())
                schedule_id_list.append(schedule_id)

        truncated_df = pd.DataFrame(np.array(truncated_rows), columns=column_labels)
        truncated_data.append(truncated_df)

    upsampled_data_combined = pd.concat(truncated_data, ignore_index=True)
    return upsampled_data_combined, time_value_list, schedule_id_list


def build_RAS_mapper(size, upsampled_data_combined, path="RAS.csv"):
    ras_df = pd.read_csv(path)
    ras_mapped = np.full((size, 2), np.nan, dtype=float)

    for i, row in upsampled_data_combined.iterrows():
        col3_val = row.iloc[3]  # IV
        col4_val = row.iloc[4]  # Digit1
        col5_val = row.iloc[5]  # Digit2
        col6_val = row.iloc[6]  # Digit3

        if col3_val == 2:
            match_r = ras_df.loc[ras_df.iloc[:, 3] == col4_val]
            if not match_r.empty:
                ras_mapped[i, 0] = match_r.iloc[0, 4]

            match_a = ras_df.loc[(ras_df.iloc[:, 0] == col5_val) & (ras_df.iloc[:, 1] == col6_val)]
            if not match_a.empty:
                ras_mapped[i, 1] = match_a.iloc[0, 2]

        elif col3_val == 1:
            match_r = ras_df.loc[ras_df.iloc[:, 8] == col4_val]
            if not match_r.empty:
                ras_mapped[i, 0] = match_r.iloc[0, 9]

            match_a = ras_df.loc[(ras_df.iloc[:, 5] == col5_val) & (ras_df.iloc[:, 6] == col6_val)]
            if not match_a.empty:
                ras_mapped[i, 1] = match_a.iloc[0, 7]

    return ras_mapped

def encode(upsampled_data_combined):
    Data = pd.DataFrame(upsampled_data_combined).copy()

    # Extract y BEFORE adding any engineered columns.
    y = Data.iloc[:, N_STATIC_COLUMNS:].values

    Data = Data.rename(
        columns={
            Data.columns[0]: "col0",  # U%
            Data.columns[1]: "col1",  # Density
            Data.columns[2]: "col2",  # Thermal_Conductivity
            Data.columns[3]: "col3",  # IV
            Data.columns[4]: "col4",  # Digit1
            Data.columns[5]: "col5",  # Digit2
            Data.columns[6]: "col6",  # Digit3
        }
    )

    for col in ["col0", "col1", "col2", "col3", "col4", "col5", "col6"]:
        Data[col] = pd.to_numeric(Data[col], errors="coerce")

    col3_map = {2: 1, 1: 3}
    Data["col3_encoded"] = Data["col3"].map(col3_map).fillna(0)

    def encode_col4(row):
        val3 = row["col3"]
        val4 = row["col4"]
        if val3 == 1:
            mapping = {1: 3, 2: 3, 5: 2, 3: 2, 4: 1}
        elif val3 == 2:
            mapping = {2: 2, 3: 2, 1: 1}
        else:
            mapping = {}
        return mapping.get(val4, 0)

    Data["col4_encoded"] = Data.apply(encode_col4, axis=1)

    encoder = OneHotEncoder(
        categories=[[1, 2, 3, 4, 5, 6]],
        sparse_output=False,
        handle_unknown="ignore",
    )
    nominal_encoded = encoder.fit_transform(Data[["col6"]])

    X_full = np.concatenate(
        [
            Data[["col0", "col1", "col5"]].fillna(0).values,
            Data[["col3_encoded", "col4_encoded"]].fillna(0).values,
            nominal_encoded,
        ],
        axis=1,
    )

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
        print("\nDEBUG AFTER load_data")
        print("combined shape:", upsampled_data_combined.shape)
        print("N_STATIC_COLUMNS:", N_STATIC_COLUMNS)
        print("len(time_value_list[0]):", len(time_value_list[0]))
        print("first 12 combined columns:", list(upsampled_data_combined.columns[:12]))
        print("last 10 combined columns:", list(upsampled_data_combined.columns[-10:]))
        ras_mapper = build_RAS_mapper(len(upsampled_data_combined), upsampled_data_combined)
        X, y = encode(upsampled_data_combined)
        X = RAS_Encode(X, ras_mapper)

        X = torch.from_numpy(np.array(X)).float()
        y = torch.from_numpy(np.array(y)).float()
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
    Create a synthetic query CSV using the same 7-static-column format as the
    real processed files. The n_u_235 argument is kept for backward-compatible
    call sites, but is not written as a static column.
    """
    if os.path.exists(path):
        os.remove(path)

    time_series = [float(step.item()) for step in resolve_time_values(t, max_len=MAX_LEN)]

    with open(path, "x", newline="") as f:
        csv_writer = csv.writer(f)
        header = [
            "U%",
            "Density",
            "Thermal_Conductivity",
            "IV",
            "Digit1",
            "Digit2",
            "Digit3",
        ]
        header.extend(time_series[:MAX_LEN])
        csv_writer.writerow(header)

        for static_position in VEHICLE_STATIC_POSITIONS[IV]:
            row = [
                U_percent,
                density,
                0,
                IV,
                static_position[0],
                static_position[1],
                static_position[2],
            ]
            row.extend([0 for _ in time_series[:MAX_LEN]])
            csv_writer.writerow(row)


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
