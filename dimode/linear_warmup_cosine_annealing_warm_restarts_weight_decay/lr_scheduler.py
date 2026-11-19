"""No-warmup cosine schedule used by the released DiMoDE commands.

Implements the original schedule's epoch-indexed learning rates without requiring
an untracked local scheduler package. Only the configuration used here is exposed.
"""
import math


class ChainedScheduler:
    def __init__(self, optimizer, T_0, T_mul=1, eta_min=0.0, last_epoch=-1,
                 max_lr=1e-4, warmup_steps=0, gamma=0.9):
        if T_0 <= 0 or T_mul != 1 or warmup_steps != 0:
            raise ValueError('This release supports T_0 > 0, T_mul=1 and warmup_steps=0.')
        self.optimizer = optimizer
        self.T_0 = T_0
        self.eta_min = eta_min
        self.max_lr = max_lr
        self.gamma = gamma
        self.last_epoch = last_epoch
        for group in optimizer.param_groups:
            group['lr'] = eta_min

    def step(self, epoch=None):
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        # The original _LRScheduler performs an initial step in its constructor.
        cycle, position = divmod(self.last_epoch + 1, self.T_0)
        peak = self.max_lr * self.gamma ** cycle
        lr = self.eta_min + (peak - self.eta_min) * (1 + math.cos(math.pi * position / self.T_0)) / 2
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
