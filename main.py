from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from config import Config as _Cfg
from models.cnn_autoencoder import CNNAutoencoder
from models.cnn_encoder import CNNEncoder
from models.mlp_autoencoder import MLPAutoencoder
from sinkhorn import sinkhorn_distance
# ======================================================================
def img_to_patches(img: torch.Tensor,
                   size: int = 128,
                   stride: int = 64) -> torch.Tensor:
    C, H, W = img.shape
    out = []
    for i in range(0, H - size + 1, stride):
        for j in range(0, W - size + 1, stride):
            out.append(img[:, i:i+size, j:j+size])
    return torch.stack(out)          # (Npatch, 3, size, size)

def train_deep_svdd(encoder: nn.Module,
                    dataloader: DataLoader,
                    *,
                    n_epoch: int = 30,
                    lr: float = 1e-4,
                    weight_decay: float = 1e-6,
                    device: torch.device = _Cfg.device):
    encoder = encoder.to(device)
    enc_opt = optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    # -- ① 初始化中心 c (均值) --
    with torch.no_grad():
        zs = []
        for x, _ in dataloader:              # dataloader 只含正常样本
            zs.append(encoder(x.to(device)))
        c = torch.mean(torch.cat(zs, 0), dim=0)
    # -- ② 训练 --
    for _ in range(n_epoch):
        encoder.train()
        for x, _ in dataloader:
            z = encoder(x.to(device))
            loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
            enc_opt.zero_grad()
            loss.backward()
            enc_opt.step()
    return c.cpu()

def evaluate_deep_svdd(encoder: nn.Module,
                       center: torch.Tensor,
                       loader: DataLoader,
                       *,
                       device: torch.device = _Cfg.device):
    encoder = encoder.to(device).eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            z = encoder(x.to(device))
            d2 = torch.sum((z - center.to(device)) ** 2, dim=1)   # 距球心平方距离
            scores.extend(d2.cpu().numpy())
            labels.extend(y.numpy())
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    roc_auc = roc_auc_score(labels != 0, scores)
    pr_auc  = average_precision_score(labels != 0, scores)
    return roc_auc, pr_auc, scores


# ======================================================================
def _load_cifar10(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    tf = transforms.ToTensor()
    ds = ConcatDataset([
        datasets.ImageFolder(str(root / "train"), transform=tf),
        datasets.ImageFolder(str(root / "test"),  transform=tf)
    ])
    X, y = [], []
    for img, cls in ds:
        X.append(img.numpy())
        y.append(0 if cls == _Cfg.NORMAL_CLASS_CIFAR10 else 1)
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64)

def _load_ecg5000(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    tr, te = root / "ECG5000_TRAIN.txt", root / "ECG5000_TEST.txt"
    if not (tr.exists() and te.exists()):
        raise FileNotFoundError("ECG5000 TXT 文件缺失")
    data = np.vstack([np.loadtxt(tr), np.loadtxt(te)])
    y = np.where(data[:, 0] == 1, 0, 1).astype(np.int64)
    X = data[:, 1:].astype(np.float32)
    return X, y

def _load_thyroid(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(root / "hypothyroid.csv")
    y = (df["binaryClass"].values == "N").astype(np.int64)  # 0=正常 1=异常
    X_raw = (df.drop(columns=["binaryClass"])
               .replace({"t":1,"f":0,"M":1,"F":0,"?":np.nan})
               .apply(pd.to_numeric, errors="coerce")
               .fillna(0)
               .values
               .astype(np.float32))
    scaler = StandardScaler().fit(X_raw[y == 0])
    return scaler.transform(X_raw).astype(np.float32), y

def _load_mvtec_cable(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    from PIL import Image
    tf = transforms.Compose([
        transforms.Resize((256, 256)),   # 如显存不足，可设 128×128
        transforms.ToTensor(),
    ])
    def _load_dir(d: Path) -> List[np.ndarray]:
        return [tf(Image.open(p).convert("RGB")).numpy()
                for p in sorted(d.glob("*.png"))]
    X, y = [], []
    train_good_dir = root / "cable" / "train" / "good"
    imgs = _load_dir(train_good_dir)
    X.extend(imgs)
    y.extend([0] * len(imgs))
    for sub in sorted((root / "cable" / "test").iterdir()):
        imgs = _load_dir(sub)
        label = 0 if sub.name == "good" else 1
        X.extend(imgs)
        y.extend([label] * len(imgs))
    return np.stack(X, dtype=np.float32), np.asarray(y, dtype=np.int64)

SETTINGS: Dict[str, Dict] = {
    "cifar10": dict(loader=_load_cifar10, model="cnn",  input_dim=None),
    "ecg5000": dict(loader=_load_ecg5000, model="mlp", input_dim=140),
    "thyroid": dict(loader=_load_thyroid, model="mlp", input_dim=None),
    "cable"  : dict(loader=_load_mvtec_cable, model="cnn", input_dim=None)
}
# ======================================================================
def _make_loaders(X: np.ndarray, y: np.ndarray, batch: int,
                  patchify: bool = False,
                  psize: int = 128, pstride: int = 64):
    if not patchify:
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        ref_ds   = TensorDataset(X_tensor[y_tensor==0], y_tensor[y_tensor==0])
        test_ds  = TensorDataset(X_tensor,             y_tensor)
        return (
            DataLoader(ref_ds,  batch_size=batch, shuffle=True,  pin_memory=True),
            DataLoader(test_ds, batch_size=batch, shuffle=False, pin_memory=True),
        )
    # ------- PATCH 模式 -------
    imgs = torch.from_numpy(X)                 # (N,3,H,W)
    labels = torch.from_numpy(y)
    X_patch, y_patch = [], []
    for img, lbl in zip(imgs, labels):
        patches = img_to_patches(img, psize, pstride)
        X_patch.append(patches)
        y_patch.extend([lbl.item()] * len(patches))
    X_patch = torch.cat(X_patch)               # (Npatch,3,psize,psize)
    y_patch = torch.tensor(y_patch)
    ref_ds  = TensorDataset(X_patch[y_patch==0], y_patch[y_patch==0])  # 仅正常 patch
    test_ds = TensorDataset(X_patch, y_patch)                          # 全部 patch
    return (
        DataLoader(ref_ds,  batch_size=batch, shuffle=True,  pin_memory=True),
        DataLoader(test_ds, batch_size=batch, shuffle=False, pin_memory=True),
    )
# 核心：带 Sinkhorn 正则的训练过程
# ======================================================================
def train_autoencoder(model: nn.Module,
                      ref_loader: DataLoader,
                      test_loader: DataLoader,
                      ckpt_name: str,
                      metrics_path: str,
                      cfg: _Cfg = _Cfg):
    device = cfg.device
    model  = model.to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    mse    = nn.MSELoss()
    center = torch.zeros(cfg.latent_dim, device=device)
    mem_Z  : Optional[torch.Tensor] = None
    history: Dict[str, List[float]] = {k: [] for k in ("epoch","loss","roc","pr")}
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        for x,_ in ref_loader:            # 仅正常样本
            x = x.to(device).float()
            recon, z = model(x)
            # 1. 重构损失
            loss_rec = mse(recon, x)
            # 2. Sinkhorn 距离正则 (z vs. center)
            loss_sink = sinkhorn_distance(
                z.detach(),                # detach 可选：若不想反传到 cost matrix
                center.expand_as(z),
                epsilon=cfg.sink_eps,
                n_iters=cfg.sink_iter,
            )
            loss = loss_rec + cfg.lam_sink * loss_sink
            optim.zero_grad()
            loss.backward()
            optim.step()
            # ------- EMA 更新中心 -------
            with torch.no_grad():
                center.mul_(1 - cfg.ema_gamma).add_(z.mean(0), alpha=cfg.ema_gamma)
            epoch_loss += loss.item() * len(x)
            # ------- 更新 memory bank-------
            if cfg.memory_size > 0:
                if mem_Z is None:
                    mem_Z = z.detach()[: cfg.memory_size]
                else:
                    mem_Z = torch.cat((mem_Z, z.detach()), dim=0)[-cfg.memory_size:]
        # ——————— 记录 & 验证 ———————
        roc, pr = _validate(model, ref_loader, test_loader, center, mem_Z, cfg)
        history["epoch"].append(epoch + 1)
        history["loss"].append(epoch_loss / len(ref_loader.dataset))
        history["roc"].append(roc)
        history["pr"].append(pr)
        print(f"[Epoch {epoch+1:3d}] loss={epoch_loss/len(ref_loader.dataset):.4f}  AUROC={roc:.4f}")
    # 保存权重 & 训练曲线
    torch.save(model.state_dict(), ckpt_name)
    with open(metrics_path, "w") as f:
        json.dump(history, f)
# 评估 & 组合分数计算
# ======================================================================
def _validate(model: nn.Module,
              ref_loader: DataLoader,
              test_loader: DataLoader,
              center: torch.Tensor,
              mem_Z: Optional[torch.Tensor],
              cfg: _Cfg):
    device = cfg.device
    model.eval()
    # 1. 参考分布 Z_ref (memory bank更贴近经验分布)
    with torch.no_grad():
        if mem_Z is not None:
            Z_ref = mem_Z.to(device)
        else:
            Z_ref = torch.cat([model(x.to(device).float())[1] for x,_ in ref_loader], dim=0)
    # 2. 主循环
    scores, labels = [], []
    mse = nn.MSELoss(reduction="none")
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device).float()
            recon, z = model(x)
            err = mse(recon, x).view(x.size(0), -1).mean(1)
            # 2a. Wasserstein 距离：点 → 经验分布(最小)
            wmin = torch.tensor([
                sinkhorn_distance(
                    zi.unsqueeze(0),
                    Z_ref,
                    epsilon=cfg.sink_eps,
                    n_iters=cfg.sink_iter,
                ).item()
                for zi in z
            ])
            score = _Cfg.ALPHA * err.cpu() + _Cfg.BETA * (-wmin.cpu())
            scores.append(score)
            labels.append(y)
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    fpr, tpr, _ = roc_curve(labels != 0, scores)
    # 若样本极不平衡，PR‑AUC 更稳定
    precision, recall, _ = roc_curve(labels, -scores)  # trick: treat score<0 as prob
    roc_auc  = auc(fpr, tpr)
    pr_auc   = auc(recall, precision)
    return roc_auc, pr_auc
# 评估可视化
# ======================================================================
def _plot_all(metrics_json: str,
              scores: np.ndarray,
              labels: np.ndarray,
              recon_errs: np.ndarray,
              latents: np.ndarray,
              title_prefix: str):
    with open(metrics_json) as f:
        hist = json.load(f)
    epochs = hist["epoch"]
    plt.figure(figsize=(12, 9))
    # 1. curve
    plt.subplot(2,2,1)
    plt.plot(epochs, hist["loss"], label="Loss")
    plt.plot(epochs, hist["roc"],  label="ROC-AUC")
    plt.plot(epochs, hist["pr"],   label="PR-AUC")
    plt.xlabel("Epoch"); plt.title("Training Dynamics"); plt.legend()
    # 2. ROC
    fpr, tpr, _ = roc_curve(labels!=0, scores)
    plt.subplot(2,2,2)
    plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--',lw=.5)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC (AUC={auc(fpr,tpr):.3f})")
    # 3. Hist
    plt.subplot(2,2,3)
    plt.hist(recon_errs[labels==0],50,alpha=.6,label="Normal")
    plt.hist(recon_errs[labels!=0],50,alpha=.6,label="Anomaly")
    plt.title("Reconstruction Error"); plt.legend()
    # 4. t‑SNE
    plt.subplot(2,2,4)
    np.random.seed(_Cfg.SEED)
    idx = np.random.choice(len(latents), size=min(1000,len(latents)), replace=False)
    tsne = TSNE(n_components=2, random_state=_Cfg.SEED).fit_transform(latents[idx])
    plt.scatter(tsne[:,0],tsne[:,1],c=labels[idx],cmap="coolwarm",s=6)
    plt.title("Latent t-SNE")
    plt.suptitle(f"{title_prefix} Results", fontsize=14)
    plt.tight_layout(rect=[0,0,1,.96])
    plt.show()
# main
# ======================================================================
def _build_model(setting: Dict) -> nn.Module:
    if setting["model"] == "cnn":
        return CNNAutoencoder(latent_dim=_Cfg.latent_dim)
    return MLPAutoencoder(input_dim=setting["input_dim"], latent_dim=_Cfg.latent_dim)

def _collect_scores(model,
                    ref_loader: DataLoader,
                    test_loader: DataLoader,
                    *,
                    use_patch: bool = False) -> Tuple[np.ndarray, ...]:
    device = _Cfg.device
    model  = model.to(device).eval()
    mse    = nn.MSELoss(reduction="none")
    # 1⃣ 收集参考潜在向量
    with torch.no_grad():
        Z_ref = torch.cat([model(x.to(device).float())[1] for x,_ in ref_loader], 0)
    scores, labels, errs, lats = [], [], [], []
    # 2⃣ 遍历测试集
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).float()
            recon, z = model(x)
            # ----- 重构误差 (patch or image) -----
            err = mse(recon, x).view(x.size(0), -1).mean(1).cpu().numpy()
            # ----- 点到分布 Wasserstein 距离 -----
            dist = np.array([
                sinkhorn_distance(
                    zi.unsqueeze(0), Z_ref,
                    epsilon=_Cfg.sink_eps, n_iters=_Cfg.sink_iter
                ).item()
                for zi in z
            ])
            score = _Cfg.ALPHA * err + _Cfg.BETA * (-dist)
            scores.extend(score)
            labels.extend(y.numpy())
            errs.extend(err)
            lats.extend(z.cpu().numpy())
    scores  = np.array(scores)
    labels  = np.array(labels)
    errs    = np.array(errs)
    latents = np.array(lats)
    return scores, labels, errs, latents

def _run_pipeline(exp_name: str, setting: Dict):
    root_map = {
        "cifar10":     _Cfg.DATA_ROOT / "cifar10",
        "ecg5000":     _Cfg.DATA_ROOT / "ECG5000",
        "thyroid":     _Cfg.DATA_ROOT,           # csv
        "cable":       _Cfg.DATA_ROOT,           # loader 内部拼 '/cable'
        "mvtec_cable": _Cfg.DATA_ROOT,
    }
    root = root_map.get(exp_name, _Cfg.DATA_ROOT)
    # ─────────── 读取数据 ───────────
    X, y = setting["loader"](root)
    if setting["input_dim"] is None:
        setting["input_dim"] = X.shape[1]
    use_patch = exp_name in {"cable", "mvtec_cable"}
    ref_loader, test_loader = _make_loaders(
        X, y, _Cfg.batch_size,
        patchify = use_patch,
        psize    = 128,
        pstride  = 64
    )
    model   = _build_model(setting)
    ckpt    = f"{exp_name}_{setting['model']}_ae.pth"
    metrics = f"{exp_name}_metrics.json"
    train_autoencoder(model, ref_loader, test_loader, ckpt, metrics)
    scores, labels, recon_errs, latents = _collect_scores(
        model, ref_loader, test_loader
    )
    if exp_name in {"cable", "mvtec_cable"}:
        from sklearn.metrics import roc_auc_score, average_precision_score
        # 1. Sinkhorn + AE
        roc_sink = roc_auc_score(labels != 0, scores)
        pr_sink  = average_precision_score(labels != 0, scores)
        # 2. AE‑Reconstruction
        roc_ae = roc_auc_score(labels != 0, recon_errs)
        pr_ae  = average_precision_score(labels != 0, recon_errs)
        # 3. Deep‑SVDD（同 CNNEncoder）
        encoder  = CNNEncoder(latent_dim=_Cfg.latent_dim).to(_Cfg.device)
        center   = train_deep_svdd(encoder, ref_loader, n_epoch=30)
        roc_svdd, pr_svdd, _ = evaluate_deep_svdd(encoder, center, test_loader)
        print("\n=== Performance Comparison (CABLE) ===")
        print("Method            | ROC-AUC | PR-AUC")
        print("------------------|---------|-------")
        print(f"Sinkhorn (ours)   | {roc_sink:.3f}  | {pr_sink:.3f}")
        print(f"AE-Reconstruction | {roc_ae :.3f}  | {pr_ae :.3f}")
        print(f"Deep SVDD         | {roc_svdd:.3f}  | {pr_svdd:.3f}\n")
    _plot_all(metrics, scores, labels, recon_errs, latents, exp_name.upper())

def main(exp_name: Optional[str] = None):
    exp_name = exp_name or _Cfg.EXPERIMENT
    if exp_name not in SETTINGS:
        raise ValueError(f"未知实验 '{exp_name}'，可选项: {list(SETTINGS)}")
    _run_pipeline(exp_name, SETTINGS[exp_name])

if __name__ == "__main__":
    import sys
    torch.manual_seed(_Cfg.SEED); random.seed(_Cfg.SEED); np.random.seed(_Cfg.SEED)
    main(sys.argv[1] if len(sys.argv)>1 else None)
