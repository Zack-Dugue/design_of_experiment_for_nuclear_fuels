import os
import glob
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy

# Scikit-Learn for Bayesian Optimization Disagreement Mapping
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# =============================================================================
# 1. MODEL ARCHITECTURE
# =============================================================================
class CausalConv1dSame(nn.Module):
    def __init__(self, cin: int, cout: int, kernel_size: int, bias: bool = True, **conv_kwargs):
        super().__init__()
        assert kernel_size >= 1
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(cin, cout, kernel_size=kernel_size, stride=1, dilation=1, padding=0, bias=bias, **conv_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)

class FFBlock(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout=0):
        super(FFBlock, self).__init__()
        self.L0 = nn.Linear(embedding_size, hidden_size)
        self.L1 = nn.Linear(hidden_size, embedding_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(embedding_size)

    def forward(self, x):
        h = self.layernorm(x)
        h = self.dropout(h)
        h = self.L0(h)
        h = self.act(h)
        h = self.L1(h)
        x = x + h
        return x

class AttentionBlock(nn.Module):
    def __init__(self, embedding_size, num_heads=8, dropout=0):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=dropout)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, batch_first=True, dropout=dropout/4)

    def forward(self, x):
        h = self.layer_norm(x)
        h = self.dropout(h)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.attention(h, h, h, attn_mask=mask, is_causal=True)[0]
        x = x + h
        return x

class FeatureEncoder(nn.Module):
    def __init__(self, num_features, embedding_dim, conv_kernel_size=16):
        super(FeatureEncoder, self).__init__()
        self.linear = nn.Linear(num_features, embedding_dim)
        self.conv = CausalConv1dSame(1, embedding_dim, kernel_size=conv_kernel_size)

    def forward(self, static_features, time_series):
        feature_mapping = self.linear(static_features)
        feature_mapping = feature_mapping.unsqueeze(1).repeat([1, time_series.size(1), 1])
        embedded_time_series = feature_mapping + torch.permute(self.conv(time_series.unsqueeze(1)), [0,2,1])
        return embedded_time_series

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, MAX_LEN=256, num_fourier_channels=None):
        super(PositionalEncoding, self).__init__()
        if num_fourier_channels is None:
            num_fourier_channels = embedding_size // 2
        self.embedding_size = embedding_size
        time_seqs = self._create_time_seqs(MAX_LEN)
        fourier_seqs = []
        for time_seq in time_seqs:
            fourier_seqs.append(self._create_fourier_seq(time_seq, num_fourier_channels))
        fourier_seqs = torch.stack(fourier_seqs, 0)
        self.register_buffer('fourier_seqs', fourier_seqs)

    def forward(self, embedding, t):
        fourier = self.fourier_seqs[:, :embedding.size(1), :]
        fourier = fourier.repeat([embedding.size(0), 1, 1, 1])
        t = t.bool().unsqueeze(-1).unsqueeze(-1)
        fourier = fourier[:,0,:,:] * (~t) + fourier[:,1,:,:] * (t)
        if fourier.size(-1) < embedding.size(-1):
            padding = torch.zeros(fourier.size(0), fourier.size(1), embedding.size(2) - fourier.size(2), device=embedding.device)
            fourier = torch.cat((fourier, padding), -1)
        return embedding + fourier

    @staticmethod
    def _create_time_seqs(MAX_LEN: int):
        MAX_LEN = MAX_LEN + 1
        seq_0_gaps = [.001, .999, 4, 5, 5, 5, 5]
        seq_0 = []
        for max_len in range((MAX_LEN // len(seq_0_gaps)) + 1):
            seq_0.extend(seq_0_gaps)
        seq_0 = seq_0[:MAX_LEN]
        seq_0 = [sum(seq_0[:i]) for i in range(len(seq_0))]
        seq_0 = seq_0[1:]
        seq_0[0] = 0

        seq_1_gaps = [.001, .999, 2, 2, 5, 5, 5, 2, 2, 2]
        seq_1 = []
        for max_len in range((MAX_LEN // len(seq_1_gaps)) + 1):
            seq_1.extend(seq_1_gaps)
        seq_1 = seq_1[:MAX_LEN]
        seq_1 = [sum(seq_1[:i]) for i in range(len(seq_1))]
        seq_1 = seq_1[1:]
        seq_1[0] = 0

        return [torch.Tensor(seq_0), torch.Tensor(seq_1)]

    @staticmethod
    def _create_fourier_seq(time_seq, num_fourier_channels: int):
        fourier_seq = torch.zeros(time_seq.shape[0], num_fourier_channels)
        for channel in range(num_fourier_channels // 2):
            for i, t in enumerate(time_seq):
                fourier_seq[i, 2*channel] = math.sin(t / (math.pow(10000, 2*channel/num_fourier_channels)))
                fourier_seq[i, 2*channel+1] = math.cos(t / (math.pow(10000, 2*channel/num_fourier_channels)))
        return fourier_seq

class StaticFeatureTransformer(nn.Module):
    def __init__(self, num_features, embedding_size, num_layers, hidden_size, num_heads, dropout=.35):
        super(StaticFeatureTransformer, self).__init__()
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.positional_encoding = PositionalEncoding(embedding_size)
        self.feature_encoder = FeatureEncoder(num_features, embedding_size)

        layer_list = nn.ModuleList([])
        for layer in range(num_layers):
            layer_list.append(FFBlock(embedding_size, hidden_size, dropout=dropout))
            layer_list.append(AttentionBlock(embedding_size, num_heads, dropout=dropout))
        self.layers = layer_list
        self.output_dropout = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(embedding_size, 1)

    def forward(self, x, t, y):
        h = self.feature_encoder(x, y)
        h = self.positional_encoding(h, t)
        for layer in self.layers:
            h = layer(h)
        return self.output_layer(h)

    def decode(self, x, t, T):
        y = torch.zeros(x.size(0), 1, 1, device=x.device)
        for tau in range(T):
            y_hat = self.forward(x, t, torch.flatten(y, 1, 2))
            y = torch.cat([y, y_hat[:, -1].unsqueeze(-1)], 1)
        return y


# =============================================================================
# 2. ENSEMBLE LOADER & MOCK GENERATOR
# =============================================================================
class SequenceEnsemble(nn.Module):
    def __init__(self, path, x_mean, x_std, y_mean, y_std, device=torch.device('cpu')):
        super(SequenceEnsemble, self).__init__()
        self.ensemble_list = nn.ModuleList([])
        self.mock_mode = False
        self.n = 5 # default if mocked

        try:
            model = torch.load(os.path.join(path, 'class_example.mdl'), map_location=device, weights_only=False)
            file_paths = glob.glob(os.path.join(path, "*.pth"))
            if len(file_paths) == 0: raise FileNotFoundError
            
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
            print(f"Failed to load models ({e}). Defaulting to MOCK mode for visualization.")
            self.mock_mode = True

        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.device = device

    @torch.no_grad()
    def member_predictions(self, x, t, T=64):
        if self.mock_mode:
            preds = []
            base_curve = torch.sin(torch.linspace(0, 3, T)) * 2.5 + 5
            base_curve += torch.linspace(0, 1, T)
            
            # Tie the mock "disagreement/variance" dynamically to the input values 
            # so the heatmap generates an interesting, varying landscape
            u_val = x[0, 0].item() if x.size(-1) > 0 else 0.9
            enr_val = x[0, 1].item() if x.size(-1) > 1 else 2.5
            landscape_multiplier = 1.0 + 3.0 * np.sin(u_val * 10) * np.cos(enr_val * 2)
            
            for _ in range(self.n):
                noise = torch.randn(T) * 0.15 * max(0.2, landscape_multiplier)
                drift = torch.cumsum(torch.randn(T) * 0.05, dim=0)
                preds.append((base_curve + noise + drift).unsqueeze(0).unsqueeze(0))
            return torch.cat(preds, dim=0).to(self.device)

        preds = []
        for model in self.ensemble_list:
            y = model.decode(x, t, T)  # [bsz, L]
            preds.append(y.unsqueeze(0))
        return torch.cat(preds, dim=0)

# =============================================================================
# 3. VIDEO / ANIMATION GENERATION
# =============================================================================
def generate_active_learning_video(
    sequence_ensemble, 
    num_iterations=10, 
    time_per_point=2.0, 
    save_path="active_learning.mp4"
):
    print("Initializing active learning simulation video...")
    
    # Setup the layout
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.06, right=0.96, wspace=0.3)
    
    # Configure the colorbar axis exactly once so the plot doesn't shrink during animation
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    
    FPS = 30
    FRAMES_PER_ITER = int(FPS * time_per_point)
    DRAW_FRAMES = int(FRAMES_PER_ITER * 0.75) 
    
    T_STEPS = 64
    n_models = sequence_ensemble.n
    frames_per_model = max(1, DRAW_FRAMES / n_models)

    # Simulate Active Learning Queries
    queries = []
    reactor_positions = ['Core-A1', 'Core-B2', 'Core-C3', 'Edge-D4', 'Center-E5']
    
    all_preds = []
    y_obj_variance = []
    
    y_mean_val = sequence_ensemble.y_mean.cpu().numpy()
    y_std_val = sequence_ensemble.y_std.cpu().numpy()

    print("Pre-evaluating ensemble across domain to establish bounds...")
    for i in range(num_iterations):
        # Using [0.8, 1.0] for Uranium Density and [0.5, 4.5] for Enrichment
        u_percent = np.random.uniform(0.8, 1.0)
        enrich_percent = np.random.uniform(0.5, 4.5)
        pos = random.choice(reactor_positions)
        queries.append((u_percent, enrich_percent, pos))
        
        # We pass the coordinates into the dummy tensor so the mock models can use them
        dummy_x = torch.zeros(1, 14).to(sequence_ensemble.device)
        dummy_x[0, 0] = u_percent
        dummy_x[0, 1] = enrich_percent
        dummy_t = torch.zeros(1).to(sequence_ensemble.device)
        
        preds = sequence_ensemble.member_predictions(dummy_x, dummy_t, T=T_STEPS)
        preds = preds.squeeze().cpu().numpy() 
        
        # Denormalize Predictions
        preds = (preds * y_std_val) + y_mean_val
        all_preds.append(preds)
        
        # Calculate ensemble disagreement objective (Standard Deviation across models, averaged over time)
        disagreement = np.mean(np.std(preds, axis=0))
        y_obj_variance.append(disagreement)
        
    actual_T_len = all_preds[0].shape[1] 
    
    # Extract the true time-step values from the model
    real_time_seqs = PositionalEncoding._create_time_seqs(actual_T_len)
    time_x = real_time_seqs[0][:actual_T_len].numpy()

    global_min_y = np.min(all_preds) * 0.95
    global_max_y = np.max(all_preds) * 1.05

    # Establish global bounds for the colorbar so it is stable across the entire video
    cbar_min = min(y_obj_variance) * 0.9
    cbar_max = max(y_obj_variance) * 1.1

    # Prepare Line Objects for Trajectories
    lines = [ax2.plot([], [], color='#1f77b4', alpha=0.5, linewidth=2)[0] for _ in range(n_models)]
    mean_line, = ax2.plot([], [], color='#d62728', linewidth=3, label='Ensemble Mean', zorder=10)
    
    ax2.set_xlim(min(time_x), max(time_x))
    ax2.set_ylim(global_min_y, global_max_y)
    ax2.set_xlabel('Time Steps', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Denormalized Prediction Value', fontweight='bold', fontsize=11)
    ax2.set_title("Ensemble Model Trajectories", fontweight='bold', fontsize=13)
    ax2.legend(loc='upper right')

    # Configure Gaussian Process Surrogate (RBF kernel)
    gp_kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
    bo_gp = GaussianProcessRegressor(kernel=gp_kernel, alpha=1e-2, normalize_y=True, optimizer='fmin_l_bfgs_b')

    # Formatter for the X-axis (0.8 -> 80%)
    def pct_formatter(x, pos):
        return f"{x*100:.0f}%"

    def generate_heatmap(ax, iter_idx):
        """Generates GP interpolation of the ensemble disagreement objective."""
        ax.clear()
        cax.clear()
        
        ax.set_xlim(0.8, 1.0)
        ax.set_ylim(0.5, 4.5)
        ax.set_xlabel('Uranium Density', fontweight='bold', fontsize=11)
        ax.set_ylabel('Enrichment %', fontweight='bold', fontsize=11)
        ax.set_title('Estimated Ensemble Disagreement Map', fontweight='bold', fontsize=13)
        ax.xaxis.set_major_formatter(FuncFormatter(pct_formatter))

        explored_pts = np.array([[q[0], q[1]] for q in queries[:iter_idx+1]])
        observed_obj = np.array(y_obj_variance[:iter_idx+1])
        
        # Grid for the heatmap
        x_grid = np.linspace(0.8, 1.0, 50)
        y_grid = np.linspace(0.5, 4.5, 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_pts = np.c_[X.ravel(), Y.ravel()]
        
        # Normalize inputs for the GP to handle scale disparity [0.8, 1.0] vs [0.5, 4.5]
        X_train_norm = np.copy(explored_pts)
        X_train_norm[:, 0] = (X_train_norm[:, 0] - 0.8) / 0.2
        X_train_norm[:, 1] = (X_train_norm[:, 1] - 0.5) / 4.0
        
        grid_pts_norm = np.copy(grid_pts)
        grid_pts_norm[:, 0] = (grid_pts_norm[:, 0] - 0.8) / 0.2
        grid_pts_norm[:, 1] = (grid_pts_norm[:, 1] - 0.5) / 4.0

        if len(X_train_norm) > 1:
            bo_gp.fit(X_train_norm, observed_obj)
            Z_pred = bo_gp.predict(grid_pts_norm)
        else:
            # First iteration, GP just predicts a flat plane at the observed value
            Z_pred = np.full(grid_pts_norm.shape[0], observed_obj[0])
            
        Z = Z_pred.reshape(X.shape)

        # Plot the Interpolated Disagreement Surface
        contour = ax.contourf(X, Y, Z, levels=np.linspace(cbar_min, cbar_max, 30), cmap='magma', extend='both')
        
        # Add colorbar specifically tied to the exact ranges
        cbar = fig.colorbar(contour, cax=cax, ticks=np.linspace(cbar_min, cbar_max, 5))
        cbar.set_label('Ensemble Variance (Obj)', fontweight='bold', fontsize=10)
        
        # Plot all previously explored points
        if iter_idx > 0:
            ax.scatter(explored_pts[:-1, 0], explored_pts[:-1, 1], c='cyan', s=45, marker='o', alpha=0.8, edgecolors='black')
        
        # Highlight the newest sampled query
        ax.scatter(explored_pts[-1, 0], explored_pts[-1, 1], c='lime', s=160, marker='*', edgecolor='black', zorder=5, label='New Query')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.8)

    def animate(frame):
        iter_idx = min(frame // FRAMES_PER_ITER, num_iterations - 1)
        sub_frame = frame % FRAMES_PER_ITER
        
        u_p, enr_p, pos = queries[iter_idx]
        preds = all_preds[iter_idx]

        # 1. Update the Disagreement Map exactly once per point iteration
        if sub_frame == 0:
            generate_heatmap(ax1, iter_idx)
            
            fig.suptitle(
                f"Active Learning Iteration: {iter_idx + 1}/{num_iterations} | "
                f"Testing U Density: {u_p*100:.1f}% | Enrichment%: {enr_p:.2f} | Pos: {pos}",
                fontsize=15, fontweight='bold', color='#222222'
            )
            
            for line in lines:
                line.set_data([], [])
            mean_line.set_data([], [])

        # 2. Draw trajectories sequentially over the real time stepping
        if sub_frame < DRAW_FRAMES:
            current_model_idx = int(sub_frame // frames_per_model)
            progress_in_model = (sub_frame % frames_per_model) / frames_per_model
            reveal_len = int(progress_in_model * actual_T_len) + 1

            for i in range(n_models):
                if i < current_model_idx:
                    lines[i].set_data(time_x, preds[i, :])
                elif i == current_model_idx:
                    lines[i].set_data(time_x[:reveal_len], preds[i, :reveal_len])
                else:
                    lines[i].set_data([], [])
        else:
            for i in range(n_models):
                lines[i].set_data(time_x, preds[i, :])
            mean_vals = np.mean(preds, axis=0)
            mean_line.set_data(time_x, mean_vals)

        return lines + [mean_line]

    total_frames = num_iterations * FRAMES_PER_ITER
    print(f"Rendering {total_frames} frames ({num_iterations} queries at {time_per_point}s each)...")
    
    ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=1000/FPS, blit=False)
    
    try:
        writer = animation.FFMpegWriter(fps=FPS, bitrate=2000)
        ani.save(save_path, writer=writer)
        print(f"Video saved successfully as {save_path}")
    except Exception as e:
        print(f"\n[NOTE]: FFmpeg not found. Automatically falling back to GIF format...")
        gif_path = save_path.replace('.mp4', '.gif')
        ani.save(gif_path, writer='pillow', fps=FPS)
        print(f"Animation saved successfully as {gif_path}")

    plt.close(fig)


if __name__ == '__main__':
    # =========================================================================
    # USER CONFIGURATION
    # =========================================================================
    NUM_POINTS = 10         # Number of total Active Learning iterations to map out
    TIME_PER_POINT = 2.0    # Duration (in seconds) to render each point
    # =========================================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Provided dataset normalization logic
    x_mean = torch.Tensor([[0.5998, 0.3474, 2.5385, 2.0769, 2.0000, 0.5435, 0.1667, 0.1667, 0.1667,
                            0.1667, 0.1667, 0.1667, 0.2849, 0.4783]])
    x_std = torch.Tensor([[0.1667, 0.4440, 0.8436, 0.7305, 0.8174, 0.4015, 0.3731, 0.3731, 0.3731,
                           0.3731, 0.3731, 0.3731, 0.3636, 0.3135]])
    y_mean = torch.Tensor([0])
    y_std = torch.Tensor([1])

    seq_ensemble = SequenceEnsemble(
        "ensembles", 
        x_mean, 
        x_std, 
        y_mean, 
        y_std, 
        device=device
    )

    generate_active_learning_video(
        sequence_ensemble=seq_ensemble, 
        num_iterations=NUM_POINTS, 
        time_per_point=TIME_PER_POINT,
        save_path="active_learning_process.mp4"
    )