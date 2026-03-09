import torch
import torch.nn as nn
import pandas as pd

from model import StaticFeatureTCN
from load_data import HGRDataset, create_synthetic_csv

from load_data import load_data
from utils import compute_n_u235
import glob
import os
from copy import deepcopy




class SequenceEnsemble(nn.Module):
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

    @torch.no_grad()
    def member_predictions(self, x, t, T=64):
        preds = []
        for model in self.ensemble_list:
            y = model.decode(x, t,T)   # expected shape [bsz, L]
            preds.append(y.unsqueeze(0))
        return torch.cat(preds, dim=0)  # [n_ensemble, bsz, L]

    def forward(self, x, t):
        preds = self.member_predictions(x, t)
        return preds.std(dim=0).mean()

    def average_over_selection(self, u_percent, density, IV, n_u_235,t, MAX_ITERS = 50, path='tmp.csv', batch_size = 32):
        create_synthetic_csv(path,u_percent,IV,density,n_u_235,t)
        averaging_dataset = HGRDataset([path],x_mean=self.x_mean,x_std=self.x_std, y_mean=self.y_mean, y_std=self.y_std)
        averaging_dataloader = torch.utils.data.DataLoader(averaging_dataset, batch_size=batch_size, shuffle=True)

        average_score = 0
        count = 0
        for (X,t,y) in averaging_dataloader:
            count += 1
            if MAX_ITERS < count:
                break
            average_score += self.member_predictions(X,t).std(0).mean()
        return average_score/count


import optuna
def make_objective(sequence_ensemble: SequenceEnsemble):
    def objective(trial):
        u_percent = trial.suggest_float("u_percent", 0.35,5)
        u_density = trial.suggest_float("u_density", 0.8,1)
        full_density = trial.suggest_float("full_density", 8,20)
        vehicle = trial.suggest_int("vehicle",1,2)
        fuel_schedule = trial.suggest_int("fuel_schedule",0,1)
        n_u_235 = compute_n_u235(u_percent,full_density,heavy_metal_fraction=u_density)

        score = sequence_ensemble.average_over_selection(u_percent,full_density,vehicle,n_u_235,fuel_schedule)
        return score
    return objective


def find_best_query(sequence_ensemble: SequenceEnsemble):
    # So here's how our Query is going to work:
    # it'll be: ()

    study = optuna.create_study(direction="maximize")
    objective = make_objective(sequence_ensemble)
    study.optimize(objective, n_trials=100)


if __name__ == '__main__':

    x_mean = torch.Tensor([[0.5998, 0.3474, 2.5385, 2.0769, 2.0000, 0.5435, 0.1667, 0.1667, 0.1667,
         0.1667, 0.1667, 0.1667, 0.2849, 0.4783]])
    x_std = torch.Tensor([[0.1667, 0.4440, 0.8436, 0.7305, 0.8174, 0.4015, 0.3731, 0.3731, 0.3731,
         0.3731, 0.3731, 0.3731, 0.3636, 0.3135]])
    y_mean =  torch.Tensor([0])
    y_std = torch.Tensor([1])
    # Use your existing loader
    model = StaticFeatureTCN(14, 256, 4, 8, dropout=0)
    seq_ensemble = SequenceEnsemble("ensembles",x_mean,x_std,y_mean,y_std)
    average_score = seq_ensemble.average_over_selection(.8, 13630,1,2.36E+20, 0)
    print(f"average_score: {average_score}")
    print("now trying optuna")
    find_best_query(sequence_ensemble=seq_ensemble)