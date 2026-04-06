import torch
from load_data import HGRDataset
from torch.utils.data import DataLoader
from model import StaticFeatureTransformer
from model import StaticFeatureTransformer, StaticFeatureTCN

import glob
import os

class MAELoss(torch.nn.Module):
    def __init__(self,reduction = 'mean'):
        super(MAELoss, self).__init__()
        pass
    def forward(self,y_hat,y):
        return torch.mean(torch.abs(y[:,1:]-y_hat[:,:-1]))

def evaluate_model(model, data_loader):
    avg_loss = 0
    # loss_fun = torch.nn.MSELoss(reduction='mean')
    loss_fun = MAELoss()
    model.eval()
    with torch.no_grad():
        for (i,(x,t,y)) in enumerate(data_loader):
            x = x.to('cuda')
            t = t.to('cuda')
            y = y.to('cuda')
            y_hat = model(x, t, y).squeeze()
            loss = loss_fun(y_hat, y)
            avg_loss += loss.item()
    #this is just to deal with train and validation splits
        if isinstance(data_loader.dataset, torch.utils.data.Subset):
            y_std = data_loader.dataset.dataset.y_std
        else:
            y_std = data_loader.dataset.y_std
    return avg_loss * y_std / len(data_loader)



def train_ensembles(save_path = "ensembles/", per_fuel_ensembles = 5):
    file_paths = glob.glob("/users/zdugue/nuclear_project/design_of_experiment_for_nuclear_fuels/fuel/*.csv")  # Adjust to your file path
    print(file_paths)
    print(f"Training {per_fuel_ensembles} models for each {len(file_paths)} held out fuel for a total of {per_fuel_ensembles*len(file_paths)} models")
    model = StaticFeatureTransformer(14, 256, 5, 512, 8, 0)
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
                continue
            train_set, interpolation_val_set = torch.utils.data.random_split(dataset, (.5, .5))

            train_loader = DataLoader(train_set,batch_size=16,shuffle=True)

            model = StaticFeatureTransformer(14, 256, 5, 512, 8, .35)
            model.cuda()
            optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0005)
            loss_fun = torch.nn.MSELoss(reduction='mean')
            eval_loss_fun = MAELoss(reduction='mean')

            lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)

            for epoch in range(300):
                model.train()
                avg_train_loss = 0
                for (j,(x,t,y)) in enumerate(train_loader):
                    x = x.to('cuda')
                    t = t.to('cuda')
                    y = y.to('cuda')

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
            extrapolation_loss = evaluate_model(model, extrapolation_val_loader)
            interpolation_val_loader = DataLoader(interpolation_val_set, batch_size=16)
            interpolation_loss = evaluate_model(model, interpolation_val_loader)
            print(f"Model complete: Final Training loss={avg_train_loss}, Final Interpolation Loss = {interpolation_loss}, Final Extrapolation Loss = {extrapolation_loss}")
            torch.save(model.state_dict(), f"{save_path}/{fuel}-{i}.pth")


if __name__ == "__main__":
    train_ensembles()