import torch
import torch.nn as nn
from .cnn_encoder import CNNEncoder

class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(CNNAutoencoder, self).__init__()
        self.encoder = CNNEncoder(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 32 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (64, 32, 32)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
