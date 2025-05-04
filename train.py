import json
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from evaluate import evaluate

def train_autoencoder(autoencoder,
                      reference_loader,
                      test_loader,
                      save_path: str,
                      metrics_path: str = "metrics.json"):
    device = Config.device
    autoencoder = autoencoder.to(device)

    criterion  = nn.MSELoss()
    optimizer  = optim.Adam(autoencoder.parameters(), lr=Config.learning_rate)

    history = {"epoch": [], "loss": [], "roc": [], "pr": []}

    for epoch in range(Config.num_epochs):
        autoencoder.train()
        total_loss = 0.0

        for batch_X, _ in reference_loader:
            batch_X = batch_X.to(device)
            optimizer.zero_grad()
            recon, _ = autoencoder(batch_X)
            loss = criterion(recon, batch_X)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 训练一轮后评估
        roc_auc, pr_auc = evaluate(autoencoder,
                                   reference_loader,
                                   test_loader,
                                   silent=True)

        history["epoch"].append(epoch + 1)
        history["loss"].append(total_loss / len(reference_loader))
        history["roc"].append(roc_auc)
        history["pr"].append(pr_auc)

        print(f"Epoch {epoch+1:3d}/{Config.num_epochs:3d} | "
              f"Loss {history['loss'][-1]:.4f} | "
              f"ROC {roc_auc:.4f} | PR {pr_auc:.4f}")

    # —— 保存模型与指标 ——
    torch.save(autoencoder.state_dict(), save_path)
    with open(metrics_path, "w") as fp:
        json.dump(history, fp, indent=2)
    print(f"模型已保存: {save_path}")
    print(f"训练曲线已保存: {metrics_path}")