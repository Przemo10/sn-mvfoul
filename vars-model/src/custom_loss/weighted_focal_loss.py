import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=2.0, alpha=1.0):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma).cuda()
        self.alpha = torch.tensor(alpha).cuda()
        self.weights = weights if weights is not None else torch.tensor([1.0])

    def forward(self, inputs, targets):
        # Ensure inputs are logits and targets are one-hot encoded
        ce_loss = F.cross_entropy(inputs, targets.argmax(dim=1), reduction='none')
        pt = torch.exp(-ce_loss)

        # Apply the weights to each class in the one-hot target
        weights = self.weights[targets.argmax(dim=1)]
        loss = weights * (1 - pt) ** self.gamma * ce_loss
        loss = loss.mean()
        return loss