import torch
import torch.nn as nn
from .mlp_encoder import MLPEncoder

class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MLPAutoencoder, self).__init__()
        self.encoder = MLPEncoder(input_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
