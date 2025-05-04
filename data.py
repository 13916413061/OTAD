import torch
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(X, y, batch_size, test_split=0.2):

    X = torch.tensor(X).float()
    y = torch.tensor(y).long()
    normal_mask = (y == 0)
    ref_dataset = TensorDataset(X[normal_mask], y[normal_mask])
    reference_loader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return reference_loader, test_loader
