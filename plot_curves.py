import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# --- Imports from your existing codebase ---
from load_data import HGRDataset, create_synthetic_csv
from utils import compute_n_u235
# Assuming this is what your top level code imports to load the model weights correctly
from model import StaticFeatureTCN 

class SequenceEnsemble(nn.Module):
    """
    Copied from your top-level code to ensure we can run this file standalone 
    without triggering the Optuna optimization from your other script.
    """
    def __init__(self, path, x_mean, x_std, y_mean, y_std, device=torch.device('cpu')):
        super(SequenceEnsemble, self).__init__()
        model = torch.load(os.path.join(path,'class_example.mdl'), map_location=device, weights_only=False)
        file_paths =  glob.glob(os.path.join(path,"*.pth"))
        self.ensemble_list = nn.ModuleList([])
        for single_path in file_paths:
            copy_model = deepcopy(model)
            state_dict = torch.load(open(single_path, "rb"))
            copy_model.load_state_dict(state_dict)
            copy_model.to(device)
            copy_model.eval()
            self.ensemble_list.append(copy_model)

        self.n = len(self.ensemble_list)
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.device = device

    @torch.no_grad()
    def member_predictions(self, x, t, T=64):
        preds =[]
        for model in self.ensemble_list:
            y = model.decode(x, t, T)   # expected shape [bsz, L]
            preds.append(y.unsqueeze(0))
        return torch.cat(preds, dim=0)  #[n_ensemble, bsz, L]

    def forward(self, x, t):
        preds = self.member_predictions(x, t)
        return preds.std(dim=0).mean()

def plot_ensemble_trajectories(
    sequence_ensemble, 
    u_percent, 
    full_density, 
    vehicle, 
    n_u_235, 
    fuel_schedule, 
    T=64, 
    save_path='ensemble_trajectories.png',
    tmp_csv_path='tmp_plot.csv'
):
    """
    Generates a synthetic dataset for the static features, predicts the time series
    for all ensemble models, and saves a beautiful plot of the trajectories.
    """
    print(f"Generating synthetic data and predicting for {len(sequence_ensemble.ensemble_list)} models...")
    
    # 1. Create the synthetic dataset with the static features
    create_synthetic_csv(
        tmp_csv_path, 
        u_percent, 
        vehicle, 
        full_density, 
        n_u_235, 
        fuel_schedule
    )

    # 2. Load it into the custom dataset using the ensemble's known scaling stats
    dataset = HGRDataset(
        [tmp_csv_path], 
        x_mean=sequence_ensemble.x_mean, 
        x_std=sequence_ensemble.x_std, 
        y_mean=sequence_ensemble.y_mean, 
        y_std=sequence_ensemble.y_std
    )

    # 3. Create a DataLoader and grab the first sample 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    X, t_batch, y = next(iter(dataloader))

    # Move to the model's device
    X = X.to(sequence_ensemble.device)
    t_batch = t_batch.to(sequence_ensemble.device)

    # 4. Generate predictions from the ensemble
    sequence_ensemble.eval()
    with torch.no_grad():
        preds = sequence_ensemble.member_predictions(X, t_batch, T=T)
    
    # Squeeze out the batch and feature dimensions -> shape [n_ensemble, seq_len]
    preds = preds.squeeze().cpu().numpy()

    # 5. Inverse transform predictions to original Y scale
    y_mean_val = sequence_ensemble.y_mean.cpu().numpy()
    y_std_val  = sequence_ensemble.y_std.cpu().numpy()
    preds_unscaled = preds * y_std_val + y_mean_val

    # 6. Set up a clean, professional plot
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass
            
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    time_steps = range(preds_unscaled.shape[1])

    # Plot all individual ensemble member trajectories
    for i in range(preds_unscaled.shape[0]):
        ax.plot(
            time_steps, 
            preds_unscaled[i, :], 
            color='#1f77b4',       # Matplotlib blue
            alpha=0.25,            # Semi-transparent to visualize density/uncertainty
            linewidth=1.5,
            zorder=2
        )

    # Plot the ensemble mean
    mean_trajectory = preds_unscaled.mean(axis=0)
    ax.plot(
        time_steps, 
        mean_trajectory, 
        color='#d62728',           # Stand-out red
        linewidth=2.5, 
        label='Ensemble Mean',
        zorder=3
    )

    # Aesthetics & Labels
    ax.set_title(
        f'Ensemble Predicted Trajectories\n'
        f'U%: {u_percent} | Density: {full_density} | Vehicle (IV): {vehicle} | '
        f'N_U_235: {n_u_235:.2e} | Schedule: {fuel_schedule}',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.set_xlabel('Time Step (T)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Value', fontsize=12, fontweight='bold')

    # Despine for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Add a subtle grid
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7, zorder=1)
    ax.legend(loc='best', fontsize=12, frameon=True, shadow=True)

    plt.tight_layout()

    # Save and clean up
    plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"Successfully saved plot to: {save_path}")

    # Remove the temporary synthetic CSV
    if os.path.exists(tmp_csv_path):
        os.remove(tmp_csv_path)

if __name__ == '__main__':
    # =========================================================================
    # 1. SET YOUR 5 STATIC ARGUMENTS HERE
    # =========================================================================
    USER_U_PERCENT = .711
    USER_U_DENSITY = 0.9      # 'heavy_metal_fraction' used to compute N_U_235
    USER_FULL_DENSITY = 13630
    USER_VEHICLE = 1          # (IV)
    USER_FUEL_SCHEDULE = 0    # (t)
    
    # Compute the derivative feature
    user_n_u_235 = compute_n_u235(
        USER_U_PERCENT, 
        USER_FULL_DENSITY, 
        heavy_metal_fraction=USER_U_DENSITY
    )
    
    # The number of steps you want to predict into the future
    PREDICTION_STEPS = 64
    # =========================================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    x_mean = torch.Tensor([[0.5998, 0.3474, 2.5385, 2.0769, 2.0000, 0.5435, 0.1667, 0.1667, 0.1667,
         0.1667, 0.1667, 0.1667, 0.2849, 0.4783]])
    x_std = torch.Tensor([[0.1667, 0.4440, 0.8436, 0.7305, 0.8174, 0.4015, 0.3731, 0.3731, 0.3731,
         0.3731, 0.3731, 0.3731, 0.3636, 0.3135]])
    y_mean = torch.Tensor([0])
    y_std = torch.Tensor([1])

    # Load the ensemble exactly as you were doing before
    # (Requires your 'ensembles' folder to be present in the same directory)
    seq_ensemble = SequenceEnsemble(
        "ensembles", 
        x_mean, 
        x_std, 
        y_mean, 
        y_std, 
        device=device
    )

    # Plot it!
    plot_ensemble_trajectories(
        sequence_ensemble=seq_ensemble,
        u_percent=USER_U_PERCENT,
        full_density=USER_FULL_DENSITY,
        vehicle=USER_VEHICLE,
        n_u_235=user_n_u_235,
        fuel_schedule=USER_FUEL_SCHEDULE,
        T=PREDICTION_STEPS, 
        save_path="ensemble_plot_output.png"
    )