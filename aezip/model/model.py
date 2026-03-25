import time
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

class SinActivation(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)

class SEncoder(torch.nn.Module):
    def __init__(self, input_dim, nlayers, latent_dim, delta, dropout=0.0, negative_slope=0.0, batch_norm=False):
        super(SEncoder, self).__init__()
        layers = []
        nunits = input_dim
        activation_fn = SinActivation()

        if isinstance(nlayers, int):
            for _ in range(nlayers):
                layers.append(torch.nn.Linear(nunits, nunits - delta))
                if batch_norm:
                    layers.append(torch.nn.BatchNorm1d(nunits - delta, eps=1e-5, track_running_stats=False))
                layers.append(activation_fn)
                layers.append(torch.nn.Dropout(dropout))
                nunits -= delta
        elif isinstance(nlayers, list):
            for layer_dim in nlayers:
                layers.append(torch.nn.Linear(nunits, layer_dim))
                if batch_norm:
                    layers.append(torch.nn.BatchNorm1d(layer_dim, eps=1e-5, track_running_stats=False))
                layers.append(activation_fn)
                layers.append(torch.nn.Dropout(dropout))
                nunits = layer_dim

        layers.append(torch.nn.Linear(nunits, latent_dim))
        layers.append(torch.nn.Tanh())
        self.encoder = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class SDecoder(torch.nn.Module):
    def __init__(self, latent_dim, nlayers, output_dim, delta, dropout=0.0, negative_slope=0.0, batch_norm=False):
        super(SDecoder, self).__init__()
        layers = []
        nunits = latent_dim
        activation_fn = SinActivation()

        if isinstance(nlayers, int):
            delta = int((output_dim - latent_dim) / (nlayers + 1))
            for _ in range(nlayers):
                layers.append(torch.nn.Linear(nunits, nunits + delta))
                if batch_norm:
                    layers.append(torch.nn.BatchNorm1d(nunits + delta, eps=1e-5, track_running_stats=False))
                layers.append(activation_fn)
                layers.append(torch.nn.Dropout(dropout))
                nunits += delta
        elif isinstance(nlayers, list):
            nlayers.reverse()
            for layer_dim in nlayers:
                layers.append(torch.nn.Linear(nunits, layer_dim))
                if batch_norm:
                    layers.append(torch.nn.BatchNorm1d(layer_dim, eps=1e-5, track_running_stats=False))
                layers.append(activation_fn)
                layers.append(torch.nn.Dropout(dropout))
                nunits = layer_dim

        layers.append(torch.nn.Linear(nunits, output_dim))
        layers.append(torch.nn.Tanh())
        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class SAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, nlayers, latent_dim, dropout=0.0, negative_slope=0.0, batch_norm=True, freeze_layers=[]):
        super(SAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if isinstance(nlayers, int):
            nlayers = nlayers - 1
            delta = int((input_dim - latent_dim) / (nlayers + 1))
        elif isinstance(nlayers, list):
            delta = None

        self.encoder = SEncoder(input_dim, nlayers, latent_dim, delta, dropout=dropout, negative_slope=negative_slope, batch_norm=batch_norm)
        self.decoder = SDecoder(latent_dim, nlayers, input_dim, delta, dropout=dropout, negative_slope=negative_slope, batch_norm=batch_norm)

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return latent, output

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
        layers.append(torch.nn.Tanh())
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
        layers.append(torch.nn.Tanh())
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
    cartesian_losses = []; angular_losses = []

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


class PeriodicLoss(torch.nn.Module):
    def __init__(self):
        """
        Loss function for periodic data like angles in the range [-180, 180].
        It computes the Mean Squared Error in the circular domain using sine and cosine transformations.
        """
        super(PeriodicLoss, self).__init__()

    def forward(self, pred, target):
        #return torch.mean(torch.cos(predictions - targets))
        #return torch.mean(abs((pred - target + 180) % 360 - 180 ))
        #pred=pred*360; target=target*360
        #print(pred[:10], target[:10])

        #ang_error = 1 - abs(torch.cos(pred - target))
        #return torch.mean(ang_error)

        delta = pred - target
        abs_delta = torch.abs(delta)
        distance = torch.min(abs_delta, 1 - abs_delta)
        return torch.mean(torch.square(distance))

class CombinedLoss(torch.nn.Module):
    """
    Custom loss function to handle Cartesian and angular data separately.

    Args:
        partition_idx (int): Index to partition the data into Cartesian and angular features.
        cartesian_weight (float): Weight for the Cartesian loss component.
        angular_weight (float): Weight for the angular loss component.
    """
    def __init__(self, partition_idx, cartesian_weight=1.0, angular_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.partition_idx = partition_idx
        self.cartesian_weight = cartesian_weight
        self.angular_weight = angular_weight

    def forward(self, output, target):
        """
        Compute the combined loss for Cartesian and angular features.

        Args:
            output (torch.Tensor): Model output (reconstructed data).
            target (torch.Tensor): Original data (ground truth).

        Returns:
            torch.Tensor: Combined loss value.
        """
        # Separate Cartesian and angular data
        cartesian_output = output[:, :self.partition_idx]
        angular_output = output[:, self.partition_idx:]
        cartesian_target = target[:, :self.partition_idx]
        angular_target = target[:, self.partition_idx:]

        # Cartesian loss (RMSD)
        cartesian_loss = torch.sqrt(torch.mean(torch.sum((cartesian_output - cartesian_target) ** 2, dim=1)))

        # Angular data loss (Cosine similarity-based in (sin, cos) space)
        angular_loss = torch.mean(1 - torch.cos(angular_output - angular_target))

        # Combine losses with respective weights
        combined_loss = cartesian_loss + angular_loss
        #combined_loss = angular_loss
        return combined_loss, cartesian_loss, angular_loss

