# config.py
from pathlib import Path

class Config:
    learning_rate = 0.0005
    batch_size = 32
    latent_dim = 16
    num_epochs = 30
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

    lam_sink   : float = 0.3
    ema_gamma  : float = 0.05           # Sinkhor 正则化系数 λ、中心 EMA γ
    memory_size: int   = 8192           # Memory-bank 设置（设为 0 关闭）

    sink_eps   : float = 0.05
    sink_iter  : int   = 20             # Sinkhorn solver 细节

    DATA_ROOT: Path = Path("data")

    EXPERIMENT: str = "cable"           # 实验名称: cifar10 | ecg5000 | thyroid | cable

    NORMAL_CLASS_CIFAR10: int = 0       # cifar10数据“正常”类别编号设置

    SEED: int = 42                      # 随机数种子

    ALPHA: float = 0.3
    BETA: float = 0.7                   # 组合分数权重
