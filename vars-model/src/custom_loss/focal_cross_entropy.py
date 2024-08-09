from torch import nn
import torch.nn.functional as F
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha = 1.0, gamma = 2.0, weight = None, reduction ='mean'):
        super(FocalCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.CE = nn.CrossEntropyLoss(reduction='none', ignore_index=-100, weight=weight)
    def forward(self, outputs, targets):
        ce_loss = nn.CrossEntropyLoss(outputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * torch.pow(1 - pt, self.gamma) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
