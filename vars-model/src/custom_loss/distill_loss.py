import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import kl_div


class DistillationLoss(nn.Module):

    def __init__(self, kld_temp=4., kld_lambda=.1):

        super(DistillationLoss, self).__init__()
        self.temp = kld_temp
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.lambda_hyperparam = kld_lambda

    def forward(self, teacher_logits, student_logits ):

        soft_targets = nn.functional.softmax(teacher_logits / self.temp, dim= 1)
        soft_prob = nn.functional.log_softmax(student_logits / self.temp, dim=1)

        kld_loss = self.kl_div(soft_prob, soft_targets)
        kld_loss = self.lambda_hyperparam * kld_loss * (self.temp ** 2)

        return kld_loss
