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
from train_ensemble import SequenceEnsemble

from model import StaticFeatureTCN

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
            score = sequence_ensemble.compute_distance((u_percent,vehicle,full_density,n_u_235,fuel_schedule),prior_point)
        return score
    return objective


def find_best_queries(sequence_ensemble: SequenceEnsemble, write_dir = 'out',n_trials =4 , num_samples = 3, gamma=1):
    write_path = Path(write_dir)
    write_path.mkdir(parents=True, exist_ok=True)
    
    

    best_queries = []
    for i in range(num_samples):
        study = optuna.create_study(direction="maximize")
        objective = make_objective(sequence_ensemble, prior_points = best_queries, gamma=gamma)
        study.optimize(objective, n_trials=n_trials)
        df = study.trials_dataframe()
        df.to_csv(write_path / f"sample_{i}_trials.csv")
        u_percent, u_density, full_density, vehicle, fuel_schedule = tuple(study.best_trials[0].params.values())
        best_queries.append((u_percent,vehicle, full_density,  compute_n_u235(u_percent,full_density,heavy_metal_fraction=u_density), fuel_schedule))
    
    with  open(write_path / "best_queries.csv", mode = 'x+') as f:
        for trial in best_queries:
            print(str(trial), file=f)


if __name__ == '__main__':

    x_mean = torch.Tensor([[0.5998, 0.3474, 2.5385, 2.0769, 2.0000, 0.5435, 0.1667, 0.1667, 0.1667,
         0.1667, 0.1667, 0.1667, 0.2849, 0.4783]])
    x_std = torch.Tensor([[0.1667, 0.4440, 0.8436, 0.7305, 0.8174, 0.4015, 0.3731, 0.3731, 0.3731,
         0.3731, 0.3731, 0.3731, 0.3636, 0.3135]])
    y_mean =  torch.Tensor([171.1579])
    y_std = torch.Tensor([105.260])
    # Use your existing loader
    model = StaticFeatureTCN(14, 256, 4, 8, dropout=0)
    seq_ensemble = SequenceEnsemble("ensembles",x_mean,x_std,y_mean,y_std,device=torch.device('cpu'))
    average_score = seq_ensemble.average_over_selection(.8, 13630,1,2.36E+20, 0)
    print(f"average_score: {average_score}")
    print("now trying optuna")
    find_best_queries(sequence_ensemble=seq_ensemble, n_trials = 4, write_dir = 'outputs/test_run_3', )
