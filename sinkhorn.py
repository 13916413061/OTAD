import torch

def sinkhorn_distance(x, y, epsilon=0.1, n_iters=50):
    C = torch.cdist(x, y, p=2).pow(2)
    batch_size = x.size(0)
    n_refs = y.size(0)
    u = torch.zeros(batch_size, device=x.device)
    v = torch.zeros(n_refs, device=x.device)
    K = torch.exp(-C / epsilon)
    for _ in range(n_iters):
        u = -epsilon * (torch.logsumexp((-C + v.unsqueeze(0)) / epsilon, dim=1))
        v = -epsilon * (torch.logsumexp((-C + u.unsqueeze(1)) / epsilon, dim=0))
    transport_cost = torch.sum(K * (C + u.unsqueeze(1) + v.unsqueeze(0)))
    return transport_cost / batch_size
