import torch
from load_data import HGRDataset
from torch.utils.data import DataLoader
from model import StaticFeatureTransformer
import glob


class MAELoss(torch.nn.Module):
    def __init__(self,reduction = 'mean'):
        super(MAELoss, self).__init__()
        pass
    def forward(self,y, y_hat):
        return torch.mean(torch.abs(y[:,1:]-y_hat[:,:-1]))

def evaluate_model(model, data_loader):
    avg_loss = 0
    # loss_fun = torch.nn.MSELoss(reduction='mean')
    loss_fun = MAELoss()
    model.eval()
    with torch.no_grad():
        for (i,(x,t,y)) in enumerate(data_loader):
            y_hat = model(x, t, y).squeeze()
            loss = loss_fun(y,y_hat)
            avg_loss += loss
    #this is just to deal with train and validation splits
        if isinstance(data_loader.dataset, torch.utils.data.Subset):
            y_std = data_loader.dataset.dataset.y_std
        else:
            y_std = data_loader.dataset.y_std
    return avg_loss * y_std / len(data_loader)



def experiment():
    file_paths = glob.glob("C:\\Users\\dugue\\Downloads\\Gustavo Code\\Code\\fuel/*.csv")  # Adjust to your file path
    print(file_paths)

    for fuel in range(len(file_paths)):
        hold_one_out_file_paths = file_paths.copy()
        hold_one_out_file_paths.pop(fuel)
        dataset = HGRDataset(hold_one_out_file_paths)
        train_set, interpolation_val_set = torch.utils.data.random_split(dataset, (.9, .1))
        extrapolation_val_set = HGRDataset([file_paths[fuel]], x_mean=dataset.x_mean, x_std=dataset.x_std, y_mean=dataset.y_mean, y_std=dataset.y_std)

        train_loader = DataLoader(train_set,batch_size=16,shuffle=True)
        interpolation_val_loader = DataLoader(interpolation_val_set,batch_size=16)
        extrapolation_val_loader = DataLoader(extrapolation_val_set, batch_size=16)

        model = StaticFeatureTransformer(14, 256, 5, 512, 8, .25)
        optimizer = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.0005)
        loss_fun = torch.nn.MSELoss(reduction='mean')
        eval_loss_fun = MAELoss(reduction='mean')

        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300)

        for epoch in range(300):
            model.train()
            avg_train_loss = 0
            lr_schedule.step()
            for (i,(x,t,y)) in enumerate(train_loader):
                optimizer.zero_grad()
                y_hat = model(x,t,y).squeeze()
                loss = loss_fun(y_hat[:,:-1],y[:,1:])
                loss.backward()
                optimizer.step()
                avg_train_loss += eval_loss_fun(y,y_hat).item()
            avg_train_loss = (avg_train_loss*train_loader.dataset.dataset.y_std)/len(train_loader)
            interpolation_val_loss = evaluate_model(model,interpolation_val_loader)
            extrapolation_val_loss = evaluate_model(model, extrapolation_val_loader)
            if epoch == 199 or epoch % 5 == 0:
                print(f"epoch = {epoch} : train_loss = {avg_train_loss}, interpolation_loss = {interpolation_val_loss}, extrapolation_loss = {extrapolation_val_loss}")



if __name__ == "__main__":
    experiment()