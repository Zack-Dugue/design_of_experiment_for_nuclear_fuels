import pandas as pd
import numpy as np
import glob
import os
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
from sklearn.preprocessing import OneHotEncoder


# Step 1: Read all CSV files
file_paths = glob.glob('C:\\Users\\dugue\\Downloads\\Gustavo Code\\Code\\fuel/*.csv')  # Adjust to your file path






def truncate_row(row, total_length=80):
    if len(row) < total_length:
        return None  # Skip rows shorter than expected
    return row[:total_length]

def load_data(file_paths):
    # Load all CSVs
    all_data = [pd.read_csv(file) for file in file_paths]

    # Define how many total columns to keep
    total_columns_to_keep = 80  # 7 features + 73 targets

    # Use one of the dataframes to define column names (if available)
    base_df = max(all_data, key=lambda df: len(df.columns))
    column_labels = list(base_df.columns[:total_columns_to_keep])

    # Truncate each row in all DataFrames
    truncated_data = []

    seq_spacing_list = []

    for df in all_data:
        truncated_rows = []
        seq_spacing_rows = []

        for row in df[df.columns].values:
            truncated_row = truncate_row(row, total_columns_to_keep)
            if truncated_row is not None:
                truncated_rows.append(truncated_row)
                seq_spacing_rows.append(26 in df.columns)
        truncated_df = pd.DataFrame(np.array(truncated_rows), columns=column_labels)
        seq_spacing_list.extend(seq_spacing_rows)
        truncated_data.append(truncated_df)

    # Combine into a single DataFrame
    upsampled_data_combined = pd.concat(truncated_data, ignore_index=True)
    # print(upsampled_data_combined)
    return upsampled_data_combined, seq_spacing_list

def build_RAS_mapper(size, upsampled_data_combined,path="RAS.csv"):
    # Read files
    # upsampled_data_combined = pd.read_csv("upsampled_data_combined.csv")
    ras_df = pd.read_csv(path)

    # Prepare array to store results (R, A)
    ras_mapped = np.full((size, 2), np.nan, dtype=float)

    # Iterate over each row
    for i, row in upsampled_data_combined.iterrows():
        col3_val = row.iloc[3]  # Vehicle ID (1 or 2)
        col4_val = row.iloc[4]
        col5_val = row.iloc[5]
        col6_val = row.iloc[6]

        if col3_val == 2:
            # --- Column 4 lookup ---
            match_r = ras_df.loc[ras_df.iloc[:, 3] == col4_val]
            if not match_r.empty:
                ras_mapped[i, 0] = match_r.iloc[0, 4]  # Assigned R

            # --- Columns 5 & 6 lookup ---
            match_a = ras_df.loc[
                (ras_df.iloc[:, 0] == col5_val) &
                (ras_df.iloc[:, 1] == col6_val)
                ]
            if not match_a.empty:
                ras_mapped[i, 1] = match_a.iloc[0, 2]  # Assigned A

        elif col3_val == 1:
            # --- Column 4 lookup ---
            match_r = ras_df.loc[ras_df.iloc[:, 8] == col4_val]
            if not match_r.empty:
                ras_mapped[i, 0] = match_r.iloc[0, 9]  # Assigned R

            # --- Columns 5 & 6 lookup ---
            match_a = ras_df.loc[
                (ras_df.iloc[:, 5] == col5_val) &
                (ras_df.iloc[:, 6] == col6_val)
                ]
            if not match_a.empty:
                ras_mapped[i, 1] = match_a.iloc[0, 7]  # Assigned A

    # Final result: NumPy array with shape (n_rows, 2)
    # print(ras_mapped)
    return ras_mapped

def encode(upsampled_data_combined):

    # 1. Load your DataFrame (already upsampled)
    Data = pd.DataFrame(upsampled_data_combined)

    # 2. Drop the first column (enumerator)
    # Data = Data.drop(columns=Data.columns[0])

    # 3. Rename the next 7 columns for clarity
    Data = Data.rename(columns={
        Data.columns[0]: 'col0',  # Nominal (2 values)
        Data.columns[1]: 'col1',  # Nominal (3 values)
        Data.columns[2]: 'col2',  # Nominal (3 values)
        Data.columns[3]: 'col3',  # Ordinal (1 > 2)
        Data.columns[4]: 'col4',  # Conditional ordinal
        Data.columns[5]: 'col5',  # Nominal (3 values)
        Data.columns[6]: 'col6',  # Nominal (6 values)
        Data.columns[7]: 'col7'
    })

    # 4. Encode col3 (ordinal: 1 > 2)
    col3_map = {2: 1, 1: 3}
    Data['col3_encoded'] = Data['col3'].map(col3_map)

    # 5. Encode col4 based on col3 (conditional ordinal)
    def encode_col4(row):
        val3 = row['col3']
        val4 = row['col4']
        if val3 == 1:
            mapping = {1: 3, 2: 3, 5: 2, 3: 2, 4: 1}
        elif val3 == 2:
            mapping = {2: 2, 3: 2, 1: 1}
        else:
            mapping = {}
        return mapping.get(val4, 0)

    def encode_col5(val):
        mapping = {1: 1, 2: 2, 3: 1}
        return mapping.get(val, 0)

    # Data['col5_encoded'] = Data['col5'].apply(encode_col5)

    Data['col4_encoded'] = Data.apply(encode_col4, axis=1)

    # 6. One-hot encode nominal columns: col0, col1, col2, col5, col6
    nominal_cols = ['col6']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    nominal_encoded = encoder.fit_transform(Data[nominal_cols])

    # 7. Concatenate all final encoded features
    X_full = np.concatenate([
        Data[['col0', 'col1', 'col5', 'col7']].values,
        Data[['col3_encoded', 'col4_encoded']].values,
        nominal_encoded,
    ], axis=1)

    # 8. Remove the first row to match your previous slicing (if needed)
    X = X_full

    # 9. Target (columns 7 and onward in the original file, adjusted due to dropped index)
    # Your targets start from the 8th original column → now column index 7 in Data
    y = Data.iloc[:, 8:80].values

    # 10. Verify dimensions
    # print(f"X shape: {X.shape}, y shape: {y.shape}")

    return X,y

def RAS_Encode(X, ras_mapped):
    column_names = ['col0', 'col1', 'col5', 'col7', 'col3_encoded', 'col4_encoded', 'feature1', 'feature2', 'feature3',
                    'feature4', 'feature5', 'feature6']
    X_full_df = pd.DataFrame(X, columns=column_names)
    X_full_df["R"] = ras_mapped[:, 0]
    X_full_df["A"] = ras_mapped[:, 1]
    scalers = {}

    for col in ['col1', 'col7', 'R', 'A']:
        if col == 'A':
            # Special scaling for A: 0 is max, far from 0 is min
            abs_A = np.abs(X_full_df[col].values.reshape(-1, 1))
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_abs = scaler.fit_transform(abs_A)
            # Invert so that closest to 0 gets 1, farthest gets 0
            X_full_df[col + '_scaled'] = 1 - scaled_abs
        else:
            scaler = MinMaxScaler()
            X_full_df   [col + '_scaled'] = scaler.fit_transform(X_full_df[[col]])

        scalers[col] = scaler
        joblib.dump(scaler, f'{col}_scaler.pkl')

    # Save the final feature matrix to CSV (optional)
    X_df = pd.DataFrame(X_full_df[['col0', 'col1_scaled', 'col3_encoded', 'col4_encoded', 'col5',
                                   'col7_scaled', 'feature1', 'feature2', 'feature3', 'feature4',
                                   'feature5', 'feature6', 'R_scaled', 'A_scaled']])

    return X_df

from torch.utils.data.dataset import Dataset

class HGRDataset(Dataset):
    def __init__(self,file_paths, x_mean=None, x_std=None, y_mean = None, y_std = None):
        super().__init__()
        upsampled_data_combined, seq_spacing_list = load_data(file_paths)
        ras_mapper = build_RAS_mapper(len(upsampled_data_combined), upsampled_data_combined)
        X, y = encode(upsampled_data_combined)
        X = RAS_Encode(X, ras_mapper)
        X = np.array(X)
        y = np.array(y)

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()


        if x_mean is None:
            x_mean =X.mean(0).unsqueeze(0)
        if x_std is None:
            x_std = X.std(0).unsqueeze(0)
        if y_mean is None:
            y_mean = y.mean().unsqueeze(0)
        if y_std is None:
            y_std = y.std().unsqueeze(0)
        X = (X - x_mean) / x_std
        y = (y - y_mean) / y_std
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std


        self.seq_spacing_list = seq_spacing_list
        self.X = X
        self.y = y


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        spacing_type = self.seq_spacing_list[idx]
        t = spacing_type

        return X,t,y



if __name__ == '__main__':
    file_paths = glob.glob("C:\\Users\\dugue\\Downloads\\Gustavo Code\\Code\\fuel/*.csv")  # Adjust to your file path
    dataset = HGRDataset(file_paths)
    X,t,y = dataset[62]
    print(f"X = {X}")
    print(f"t = {t}")
    print(f"y = {y}")

