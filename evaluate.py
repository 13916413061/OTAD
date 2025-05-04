import torch
import numpy as np
from config import Config
from sinkhorn import sinkhorn_distance
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate(autoencoder, reference_loader, test_loader, *, silent: bool = False):
    device = Config.device
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    # ① 收集参考潜在向量
    with torch.no_grad():
        ref_z = torch.cat([
            autoencoder(batch_X.to(device))[1]   # 只要 z
            for batch_X, _ in reference_loader
        ], dim=0)

    # ② 计算每个测试样本与参考集的 Sinkhorn 距离
    scores, labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            z_batch = autoencoder(batch_X.to(device))[1]
            for z_i, y_i in zip(z_batch, batch_y):
                dist = sinkhorn_distance(z_i.unsqueeze(0), ref_z)
                scores.append(dist.item())
                labels.append(y_i.item())

    scores, labels = np.array(scores), np.array(labels)
    roc_auc = roc_auc_score(labels != 0, scores)
    pr_auc  = average_precision_score(labels != 0, scores)

    if not silent:
        print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    return roc_auc, pr_auc
