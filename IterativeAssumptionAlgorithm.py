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


from model import StaticFeatureTCN
from load_data import HGRDataset, create_synthetic_csv

from load_data import load_data
from utils import compute_n_u235
import glob
import os
from copy import deepcopy
import optuna
from pathlib import Path

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
        preds = []
        for model in self.ensemble_list:
            y = model.decode(x, t, T)  # [bsz, L]
            preds.append(y.unsqueeze(0))
        return torch.cat(preds, dim=0)
    
    def forward(self, x,t,T=100):
        return self.member_predictions(x,t,T=T).variance(dim=0).mean()


    def average_over_selection(self, u_percent, density, IV, n_u_235,t, MAX_ITERS = 50, path='tmp.csv', batch_size = 32):
        create_synthetic_csv(path,u_percent,IV,density,n_u_235,t)
        averaging_dataset = HGRDataset([path],x_mean=self.x_mean,x_std=self.x_std, y_mean=self.y_mean, y_std=self.y_std)
        averaging_dataloader = torch.utils.data.DataLoader(averaging_dataset, batch_size=batch_size, shuffle=True)

        average_score = 0
        count = 0
        for (X,t,y) in averaging_dataloader:
            X,t,y = X.to(self.device), t.to(self.device), y.to(self.device)
            count += 1
            if MAX_ITERS < count:
                break
            average_score += self.member_predictions(X,t).var(0).mean()
        return average_score/count

    
    @staticmethod
    def cheap_compute_distance(y_1,y_2):
        distance = ((y_1 - y_2)**2).mean()
        return distance    


    def compute_distance(self, X_1, X_2,MAX_ITERS = 50, path='tmp.csv', batch_size = 32):
        u_percent,IV,density,n_u_235,t = X_1
        create_synthetic_csv(path,u_percent,IV,density,n_u_235,t)
        averaging_dataset_1 = HGRDataset([path],x_mean=self.x_mean,x_std=self.x_std, y_mean=self.y_mean, y_std=self.y_std)
        averaging_dataloader_1 = torch.utils.data.DataLoader(averaging_dataset, batch_size=batch_size, shuffle=False)
        u_percent,IV,density,n_u_235,t = X_2
        create_synthetic_csv(path,u_percent,IV,density,n_u_235,t)
        averaging_dataset_2 = HGRDataset([path],x_mean=self.x_mean,x_std=self.x_std, y_mean=self.y_mean, y_std=self.y_std)
        averaging_dataloader_2 = torch.utils.data.DataLoader(averaging_dataset, batch_size=batch_size, shuffle=False)
        distance = 0
        count = 0
        for ((x_1, t_1, _),(x_2, t_2, _)) in averaging_dataloader_1:
            x_1,t_1 = x_1.to(self.device), t_1.to(self.device)
            x_2,t_2 = x_2.to(self.device), t_2.to(self.device),
            y_1 = self.member_predictions(x_1,t_1)
            y_2 = self.member_predictions(x_2,t_2)
            length = min(y_1.size(1), y_2.size(1))
            count += 1
            if MAX_ITERS < count:
                break
            distance += self.cheap_compute_distance(y_1[:,:length] , y_2[:,:length])
        return average_score/count





def make_objective(sequence_ensemble: SequenceEnsemble, prior_points : list[torch.Tensor], gamma=1):
    def objective(trial):
        u_percent = trial.suggest_float("u_percent", 0.35,5)
        u_density = trial.suggest_float("u_density", 0.8,1)
        full_density = trial.suggest_float("full_density", 9000,16000)
        vehicle = trial.suggest_int("vehicle",1,2)
        fuel_schedule = trial.suggest_int("fuel_schedule",0,1)
        n_u_235 = compute_n_u235(u_percent,full_density,heavy_metal_fraction=u_density)

        score = sequence_ensemble.average_over_selection(u_percent,full_density,vehicle,n_u_235,fuel_schedule)
        for prior_point in prior_points:
            score = sequence_ensemble.compute_distance((u_percent,u_density,full_density,fuel_schedule,n_u_235),)
        return score
    return objective


def find_best_queries(sequence_ensemble: SequenceEnsemble, write_dir = 'out',n_trials =4 , num_samples = 3, gamma=1):
    write_path = Path(write_dir)
    write_path.mkdir(parents=True, exist_ok=True)
    
    

    best_queries = []
    for i in range(num_samples):
        study = optuna.create_study(direction="maximize")
        objective = make_objective(sequence_ensemble, prior_points = [], gamma=gamma)
        study.optimize(objective, n_trials=n_trials)
        df = study.trials_dataframe()
        df.to_csv(write_path / f"sample_{i}_trials.csv")
        best_queries.append(study.best_trials[0])
    
    with  open(write_path / "best_queries.csv", mode = 'x+') as f:
        for trial in best_queries:
            print(str(trial), file=f)


if __name__ == '__main__':

    x_mean = torch.Tensor([[0.5998, 0.3474, 2.5385, 2.0769, 2.0000, 0.5435, 0.1667, 0.1667, 0.1667,
         0.1667, 0.1667, 0.1667, 0.2849, 0.4783]])
    x_std = torch.Tensor([[0.1667, 0.4440, 0.8436, 0.7305, 0.8174, 0.4015, 0.3731, 0.3731, 0.3731,
         0.3731, 0.3731, 0.3731, 0.3636, 0.3135]])
    y_mean =  torch.Tensor([0])
    y_std = torch.Tensor([1])
    # Use your existing loader
    model = StaticFeatureTCN(14, 256, 4, 8, dropout=0)
    seq_ensemble = SequenceEnsemble("ensembles",x_mean,x_std,y_mean,y_std,device=torch.device('cuda'))
    average_score = seq_ensemble.average_over_selection(.8, 13630,1,2.36E+20, 0)
    print(f"average_score: {average_score}")
    print("now trying optuna")
    find_best_queries(sequence_ensemble=seq_ensemble, n_trials = 4, write_dir = 'outputs/test_run_2')
