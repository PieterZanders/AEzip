import time
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, nlayers, latent_dim, delta, dropout=0.0, negative_slope=0.0, batch_norm=False):
        super(Encoder, self).__init__()
        layers = []
        nunits = input_dim

        if isinstance(nlayers, int):
            for _ in range(nlayers):
                layers.append(torch.nn.Linear(nunits, nunits - delta))
                if batch_norm:
                    layers.append(torch.nn.BatchNorm1d(nunits - delta, eps=1e-5, track_running_stats=False))
                layers.append(torch.nn.LeakyReLU(negative_slope))
                layers.append(torch.nn.Dropout(dropout))
                nunits -= delta
        elif isinstance(nlayers, list):
            for layer_dim in nlayers:
                layers.append(torch.nn.Linear(nunits, layer_dim))
                if batch_norm:
                    layers.append(torch.nn.BatchNorm1d(layer_dim, eps=1e-5, track_running_stats=False))
                layers.append(torch.nn.LeakyReLU(negative_slope))
                layers.append(torch.nn.Dropout(dropout))
                nunits = layer_dim

        layers.append(torch.nn.Linear(nunits, latent_dim))
        layers.append(torch.nn.Sigmoid())
        self.encoder = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, nlayers, output_dim, delta, dropout=0.0, negative_slope=0.0, batch_norm=False):
        super(Decoder, self).__init__()
        layers = []
        nunits = latent_dim

        if isinstance(nlayers, int):
            delta = int((output_dim - latent_dim) / (nlayers + 1))
            for _ in range(nlayers):
                layers.append(torch.nn.Linear(nunits, nunits + delta))
                if batch_norm:
                    layers.append(torch.nn.BatchNorm1d(nunits + delta, eps=1e-5, track_running_stats=False))
                layers.append(torch.nn.LeakyReLU(negative_slope))
                layers.append(torch.nn.Dropout(dropout))
                nunits += delta
        elif isinstance(nlayers, list):
            nlayers.reverse()
            for layer_dim in nlayers:
                layers.append(torch.nn.Linear(nunits, layer_dim))
                if batch_norm:
                    layers.append(torch.nn.BatchNorm1d(layer_dim, eps=1e-5, track_running_stats=False))
                layers.append(torch.nn.LeakyReLU(negative_slope))
                layers.append(torch.nn.Dropout(dropout))
                nunits = layer_dim

        layers.append(torch.nn.Linear(nunits, output_dim))
        layers.append(torch.nn.Sigmoid())
        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, nlayers, latent_dim, dropout=0.0, negative_slope=0.0, batch_norm=False, freeze_layers=[]):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if isinstance(nlayers, int):
            nlayers = nlayers - 1
            delta = int((input_dim - latent_dim) / (nlayers + 1))
        elif isinstance(nlayers, list):
            delta = None

        self.encoder = Encoder(input_dim, nlayers, latent_dim, delta, dropout=dropout, negative_slope=negative_slope, batch_norm=batch_norm)
        self.decoder = Decoder(latent_dim, nlayers, input_dim, delta, dropout=dropout, negative_slope=negative_slope, batch_norm=batch_norm)

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return latent, output

def compress(model, train_dataloader, optimizer, loss_function, epochs, step_size, gamma, device):
    train_losses = []; compressed = []

    # Initialize scheduler
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    print("Start Training:")
    for epoch in range(epochs):
        start_time = time.time()

        train_losses = []
        model.train()
        for data in train_dataloader:
            data = data.to(device)
            latent, output = model(data)
            loss = loss_function(output, data)  # , cartesian_loss, angular_loss
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            #cartesian_losses.append(cartesian_loss.item())
            #angular_losses.append(angular_loss.item())

            if epoch == epochs-1:
                compressed.append(latent.detach().cpu().numpy())

        avg_train_loss = torch.mean(torch.tensor(train_losses)).item()
        #avg_cartesian_loss = torch.mean(torch.tensor(cartesian_losses)).item()
        #avg_angular_loss = torch.mean(torch.tensor(angular_losses)).item()
        train_losses.append(avg_train_loss)
        
        # Step the scheduler
        scheduler.step()

        end_time = time.time()
        epoch_duration = end_time - start_time

        if (epoch) % 1 == 0 or epoch == epochs-1:
            print(f'Epoch {epoch+1}/{epochs} | Total Loss: {avg_train_loss:.6f} ({(90.0 - np.degrees(np.arccos(avg_train_loss))):.2f}) | lr: {sum(scheduler.get_last_lr()):.2e} | Duration: {epoch_duration:.2f}s')  
            # | XYZLoss: {avg_cartesian_loss:.4f} | DIHLoss: {avg_angular_loss:.4f}
    

    return train_losses, np.concatenate(compressed)

def decompress(model, dataloader, device):
    model.eval()
    x_hat = []
    for data in dataloader:
        data = data.to(device)
        output = model(data)
        x_hat.append(output.detach().cpu().numpy())
    return np.concatenate(x_hat, axis=0)

