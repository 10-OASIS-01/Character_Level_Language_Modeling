# utils/helpers.py

import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def evaluate(model, dataset, device, batch_size=50):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    losses = []
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        logits, loss = model(X, Y)
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    return mean_loss
