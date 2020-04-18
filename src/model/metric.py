import numpy as np
import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def avg_earliest_incorrect(output, target, iters=10):
    B = output.shape[0] // iters
    pred = torch.argmax(output, dim=1)
    earliest_incorrect = np.array([iters + 1] * B)
    for i in reversed(range(iters)):
        pred_ = pred[i * B: (i + 1) * B].cpu().numpy()
        target_ = target[i * B: (i + 1) * B].cpu().numpy()
        incorrect = np.argwhere(pred_ != target_)
        earliest_incorrect[incorrect] = i + 1
    return np.mean(earliest_incorrect)
