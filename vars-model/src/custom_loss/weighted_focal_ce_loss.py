from torch import nn
import torch.nn.functional as F
import torch
from .focal_cross_entropy import FocalLoss



class WeightedFocalCELoss(nn.Module):
    def __init__(self, alpha = 1.0, gamma = 2.0, ce_weight = 0.75,  weight = None):
        super(WeightedFocalCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce_weight = ce_weight
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100, weight=weight)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, weight=None, reduction='mean')
    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        ce_loss = self.cross_entropy(inputs, targets)

        loss = self.ce_weight * ce_loss + (1- self.ce_weight) * focal_loss

        return loss


