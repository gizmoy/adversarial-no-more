import torch
import torch.nn.functional as F


def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)


def harmonic_mean_loss(output, eps=1e-6):
    B, C = output.shape
    probs = F.softmax(output, dim=1)
    batch_losses = - C / torch.sum((probs + eps).pow(-1), dim=1)
    return C * torch.sum(batch_losses) / B


def proposed_loss(output, target, beta):
    return cross_entropy_loss(output, target) + beta * harmonic_mean_loss(output)
