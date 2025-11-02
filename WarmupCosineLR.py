import warnings
import math
from typing import List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, max_epochs: int, warmup_start_lr: float = 1e-8, eta_min: float = 1e-8, last_epoch: int = -1,):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("Use `get_last_lr()` instead of `get_lr()` directly.", UserWarning)

        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]
    
    def _get_closed_form_lr(self) -> List[float]:
        return self.get_lr()
if __name__ == "__main__":
    print ("Schedule ran")