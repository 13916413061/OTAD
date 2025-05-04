import torch.nn as nn
from models.cnn_encoder import CNNEncoder
import torch.nn.functional as F
class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # ----- Encoder: 见之前给的 ResNet-style CNNEncoder -----
        self.encoder = CNNEncoder(latent_dim)   # 只输出 z

        # ----- Decoder -----
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256*16*16), nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(),  # 16→32
            nn.ConvTranspose2d(128,64,4,2,1),  nn.ReLU(),  # 32→64
            nn.ConvTranspose2d(64,32,4,2,1),   nn.ReLU(),  # 64→128
            nn.ConvTranspose2d(32,3,4,2,1),    nn.Sigmoid()# 128→256
        )

    def forward(self, x):
        z = self.encoder(x)                        # (B, latent_dim)   或 (B,C,32,32) for方案A
        y = self.fc(z).view(-1,256,16,16)          # ← 你的 decoder 前半部分
        y = self.deconv(y)                         # 默认 256×256

        # ★ 动态调整到与输入同尺寸 ★
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return y, z

