import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MLPEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.net(x)
