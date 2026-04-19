import torch
from load_data import HGRDataset, create_synthetic_csv
from torch.utils.data import DataLoader
from model import StaticFeatureTransformer
from model import StaticFeatureTransformer, StaticFeatureTCN
import torch.nn as nn
from copy import deepcopy
import glob
import os

class MAELoss(torch.nn.Module):
    def __init__(self,reduction = 'mean'):
        super(MAELoss, self).__init__()
        pass
    def forward(self,y_hat,y):
        return torch.mean(torch.abs(y[:,1:]-y_hat[:,:-1]))

class SequenceEnsemble(nn.Module):
    def __init__(self, path, x_mean, x_std, y_mean, y_std, device=torch.device('cpu')):
        super(SequenceEnsemble, self).__init__()
        self.ensemble_list = nn.ModuleList([])
        self.mock_mode = False
        self.n = 5  # default if mocked

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
            print(f"Failed to load models ({e}).")
            assert False

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

    def forward(self, x, t, T=100):
        return self.member_predictions(x, t, T=T).variance(dim=0).mean()

    def average_over_selection(self, u_percent, IV,density, n_u_235, t, MAX_ITERS=50, path='tmp.csv', batch_size=32):
        create_synthetic_csv(path, u_percent, IV, density, n_u_235, t)
        averaging_dataset = HGRDataset([path], x_mean=self.x_mean, x_std=self.x_std, y_mean=self.y_mean,
                                       y_std=self.y_std)
        averaging_dataloader = torch.utils.data.DataLoader(averaging_dataset, batch_size=batch_size, shuffle=True)

        average_score = 0
        count = 0
        for (X, t, y) in averaging_dataloader:
            X, t, y = X.to(self.device), t.to(self.device), y.to(self.device)
            count += 1
            if MAX_ITERS < count:
                break
            average_score += self.member_predictions(X, t).var(0).mean()
        return average_score / count

    @staticmethod
    def cheap_compute_distance(y_1, y_2):
        distance = ((y_1 - y_2) ** 2).mean()
        return distance
    def compute_distance(self, X_1, X_2, MAX_ITERS=50, path='tmp.csv', batch_size=32, simple=True):
        with torch.no_grad():
            u_percent, IV, density, n_u_235, t = X_1
            create_synthetic_csv(path, u_percent, IV,density, n_u_235, t)
            averaging_dataset_1 = HGRDataset([path], x_mean=self.x_mean, x_std=self.x_std, y_mean=self.y_mean,
                                             y_std=self.y_std)
            averaging_dataloader_1 = torch.utils.data.DataLoader(averaging_dataset_1, batch_size=batch_size, shuffle=False)
            u_percent, IV, density, n_u_235, t = X_2
            create_synthetic_csv(path, u_percent,IV,density, n_u_235, t)
            averaging_dataset_2 = HGRDataset([path], x_mean=self.x_mean, x_std=self.x_std, y_mean=self.y_mean,
                                             y_std=self.y_std)
            averaging_dataloader_2 = torch.utils.data.DataLoader(averaging_dataset_2, batch_size=batch_size, shuffle=False)
            distance = 0
            count = 0
            for ((x_1, t_1, _), (x_2, t_2, _)) in zip(averaging_dataloader_1, averaging_dataloader_2):
                x_1, t_1 = x_1.to(self.device), t_1.to(self.device)
                x_2, t_2 = x_2.to(self.device), t_2.to(self.device),
                if simple:
                    distance += ((x_1[:4] - x_2[:4])**2).mean()
                else:
                    y_1 = self.member_predictions(x_1, t_1)
                    y_2 = self.member_predictions(x_2, t_2)
                    length = min(y_1.size(1), y_2.size(1))
                    distance += self.cheap_compute_distance(y_1[:, :length], y_2[:, :length])
                count += 1
                if MAX_ITERS < count:
                    break
        return distance / count



def evaluate_model(model, data_loader, device=torch.device('cuda')):
    avg_loss = 0
    # loss_fun = torch.nn.MSELoss(reduction='mean')
    loss_fun = MAELoss()
    model.eval()
    with torch.no_grad():
        for (i,(x,t,y)) in enumerate(data_loader):
            x = x.to(device)
            t = t.to(device)
            y = y.to(device)
            y_hat = model(x, t, y).squeeze()
            loss = loss_fun(y_hat, y)
            avg_loss += loss.item()
    #this is just to deal with train and validation splits
        if isinstance(data_loader.dataset, torch.utils.data.Subset):
            y_std = data_loader.dataset.dataset.y_std
        else:
            y_std = data_loader.dataset.y_std
    return avg_loss * y_std / len(data_loader)



def train_ensembles(save_path = "ensembles/", file_paths = glob.glob("fule/" + "*.csv"), per_fuel_ensembles = 5, device=torch.device('cuda'), T=300, overwrite_file = False):
    print(file_paths)
    print(f"Training {per_fuel_ensembles} models for each {len(file_paths)} held out fuel for a total of {per_fuel_ensembles*len(file_paths)} models")
    model = StaticFeatureTransformer(14, 256, 1, 512, 8, 0)
    if not os.path.exists(f"{save_path}"):
        os.makedirs(f"{save_path}")
    torch.save(model, os.path.join(save_path, f"class_example.mdl"))
    for fuel in range(len(file_paths)):
        print(f"Onto fuel: {fuel}")
        hold_one_out_file_paths = file_paths.copy()
        hold_one_out_file_paths.pop(fuel)
        dataset = HGRDataset(hold_one_out_file_paths)
        for i in range(per_fuel_ensembles):
            print(f"\nGenerating Ensemble : {fuel}-{i}")
            if os.path.isfile(f"{save_path}/{fuel}-{i}.pth"):
                print("This model has already been trained.")
                if overwrite_file:
                    os.remove(f"{save_path}/{fuel}-{i}.pth")
                else:
                    continue
            train_set, interpolation_val_set = torch.utils.data.random_split(dataset, (.5, .5))

            train_loader = DataLoader(train_set,batch_size=16,shuffle=True)

            model = StaticFeatureTransformer(14, 256, 1, 512, 8, 0)
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0005)
            loss_fun = torch.nn.MSELoss(reduction='mean')
            eval_loss_fun = MAELoss(reduction='mean')

            lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T)

            for epoch in range(T):
                model.train()
                avg_train_loss = 0
                for (j,(x,t,y)) in enumerate(train_loader):
                    x = x.to(device)
                    t = t.to(device)
                    y = y.to(device)

                    optimizer.zero_grad()
                    y_hat = model(x,t,y).squeeze()
                    loss = loss_fun(y_hat[:,:-1],y[:,1:])
                    loss.backward()
                    optimizer.step()
                    avg_train_loss += eval_loss_fun(y_hat,y).item()
                avg_train_loss = (avg_train_loss*train_loader.dataset.dataset.y_std)/len(train_loader)
                lr_schedule.step()

                if epoch == 299 or epoch % 5 == 0:
                    print(f"epoch = {epoch} : train_loss = {avg_train_loss} \r", end="")
            extrapolation_val_set = HGRDataset([file_paths[fuel]], x_mean=dataset.x_mean, x_std=dataset.x_std, y_mean=dataset.y_mean, y_std=dataset.y_std)
            extrapolation_val_loader = DataLoader(extrapolation_val_set, batch_size=16)
            extrapolation_loss = evaluate_model(model, extrapolation_val_loader,device=device)
            interpolation_val_loader = DataLoader(interpolation_val_set, batch_size=16)
            interpolation_loss = evaluate_model(model, interpolation_val_loader,device=device)
            print(f"Model complete: Final Training loss={avg_train_loss}, Final Interpolation Loss = {interpolation_loss}, Final Extrapolation Loss = {extrapolation_loss}")
            torch.save(model.state_dict(), f"{save_path}/{fuel}-{i}.pth")



if __name__ == "__main__":
    train_ensembles(device=torch.device('cpu'),per_fuel_ensembles=1, T=50)
