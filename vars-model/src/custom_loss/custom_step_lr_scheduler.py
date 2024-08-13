import torch
import numpy as np

class CustomStepLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, step_size=3, gamma=0.9, k=3, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.n_epochs = step_size * k -1
        # Parameters that change after n_epochs
        self.gamma_increase_factor = np.round(np.sqrt(gamma)/ gamma,2)
        self.step_size_increase = 1

        # Initialize the parent class
        super(CustomStepLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # For the first n epochs, use the standard StepLR behavior
        if self.last_epoch < self.n_epochs:
            if  self.last_epoch > 0 and self.last_epoch % self.step_size == 0:
                return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
            else:
                return [group['lr'] for group in self.optimizer.param_groups]

        # After n_epochs, increase gamma and step_size
        if self.last_epoch == self.n_epochs:
            print(self.last_epoch, self.n_epochs)
            self.gamma *= self.gamma_increase_factor
            self.step_size += self.step_size_increase
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self.last_epoch == self.n_epochs + 1:
            return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


        # Apply the updated gamma and step_size after n_epochs
        if (self.last_epoch +1) % self.step_size == 0:
            return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

        return [group['lr'] for group in self.optimizer.param_groups]
