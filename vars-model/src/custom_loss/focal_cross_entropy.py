from torch import nn
import torch.nn.functional as F
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha = 1.0, gamma = 2.0, weight = None, reduction ='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma).cuda()
        self.alpha = torch.tensor(alpha).cuda()
        self.reduction = reduction
        self.CE = nn.CrossEntropyLoss(reduction='none',reduce=False, ignore_index=-100, weight=weight)
    def forward(self, outputs, targets):
        ce_loss = self.CE(outputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * torch.pow(1 - pt, self.gamma) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
