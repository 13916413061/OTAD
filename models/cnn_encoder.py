import torch
import torch.nn as nn
import torch.nn.functional as F

class _Block(nn.Module):
    """简单 2×Conv 残差块"""
    def __init__(self, ch_in, ch_out, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch_out)

        self.skip  = nn.Sequential()
        if downsample or ch_in != ch_out:
            self.skip = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 1, stride, bias=False),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        return F.relu(out)

class CNNEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.stem = nn.Sequential(                 # 256 → 128
            nn.Conv2d(3, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.layer1 = _Block(32,  64, downsample=True)   # 128 → 64
        self.layer2 = _Block(64, 128, downsample=True)   # 64  → 32
        self.layer3 = _Block(128,256, downsample=True)   # 32  → 16
        self.pool   = nn.AdaptiveAvgPool2d(1)            # H,W 任意 → 1×1
        self.fc     = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)      # (B,256)
        return self.fc(x)
