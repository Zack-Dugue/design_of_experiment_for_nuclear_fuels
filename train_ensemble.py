import torch
from load_data import HGRDataset
from torch.utils.data import DataLoader
from model import StaticFeatureTransformer
import glob
import os

class MAELoss(torch.nn.Module):
    def __init__(self,reduction = 'mean'):
        super(MAELoss, self).__init__()
        pass
    def forward(self,y, y_hat):
        return torch.mean(torch.abs(y-y_hat))

def evaluate_model(model, data_loader):
    avg_loss = 0
    # loss_fun = torch.nn.MSELoss(reduction='mean')
    loss_fun = MAELoss()
    model.eval()
    with torch.no_grad():
        for (i,(x,t,y)) in enumerate(data_loader):
            y_hat = model(x, t, y).squeeze()
            loss = loss_fun(y_hat, y)
            avg_loss += loss
    #this is just to deal with train and validation splits
        if isinstance(data_loader.dataset, torch.utils.data.Subset):
            y_std = data_loader.dataset.dataset.y_std
        else:
            y_std = data_loader.dataset.y_std
    return avg_loss * y_std / len(data_loader)



def train_ensembles(save_path = "ensembles/", per_fuel_ensembles = 1, ):
    file_paths = glob.glob("C:\\Users\\dugue\\Downloads\\Gustavo Code\\Code\\fuel/*.csv")  # Adjust to your file path
    print(file_paths)
    model = StaticFeatureTransformer(14, 256, 3, 512, 8, 0)
    torch.save(model, os.path.join(save_path, f"class_example.mdl"))
    for fuel in range(len(file_paths)):

        hold_one_out_file_paths = file_paths.copy()
        hold_one_out_file_paths.pop(fuel)
        dataset = HGRDataset(hold_one_out_file_paths)
        model = StaticFeatureTransformer(14, 256, 3, 512, 8, 0)

        for i in range(per_fuel_ensembles):
            train_set, interpolation_val_set = torch.utils.data.random_split(dataset, (.5, .5))

            train_loader = DataLoader(train_set,batch_size=16,shuffle=True)


            optimizer = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.001)
            loss_fun = torch.nn.MSELoss(reduction='mean')
            eval_loss_fun = MAELoss(reduction='mean')

            lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200)

            for epoch in range(200):
                model.train()
                avg_train_loss = 0
                lr_schedule.step()
                for (j,(x,t,y)) in enumerate(train_loader):
                    optimizer.zero_grad()
                    y_hat = model(x,t,y).squeeze()
                    loss = loss_fun(y_hat,y)
                    loss.backward()
                    optimizer.step()
                    avg_train_loss += eval_loss_fun(y_hat,y).item()
                avg_train_loss = (avg_train_loss*train_loader.dataset.dataset.y_std)/len(train_loader)

                if epoch == 99 or epoch % 5 == 0:
                    print(f"epoch = {epoch} : train_loss = {avg_train_loss}")
            torch.save(model.state_dict(), os.path.join(save_path, f"{fuel}-{i}.pth"))
            print(f"model saved to {os.path.join(save_path, f"{fuel}-{i}.pth")}")


if __name__ == "__main__":
    train_ensembles()