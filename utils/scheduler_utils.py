import torch

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """
    Gradually warm-up(increasing) learning rate in optimizer proposed in
    Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour [Goyal et al., 2017].

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if
            multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends
            up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler (e.g.,
            ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        wrapper_state_dict = {key: value for key, value in self.__dict__.items() if (key != 'optimizer' and key !='after_scheduler')}
        wrapped_state_dict = {key: value for key, value in self.after_scheduler.__dict__.items() if key != 'optimizer'} 
        return {'wrapped': wrapped_state_dict, 'wrapper': wrapper_state_dict}
    
    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict['wrapper'])
        self.after_scheduler.__dict__.update(state_dict['wrapped'])

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True

                return self.after_scheduler.get_last_lr()

            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        # ReduceLROnPlateau is called at the end of epoch, whereas others are
        # called at beginning
        self.last_epoch = epoch if epoch != 0 else 1

        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                for base_lr in self.base_lrs
            ]

            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)

                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class LRPolicy(object):
    def __init__(self, lr_decay, decay_step, lowest_decay):
        self.lr_decay = lr_decay
        self.decay_step = decay_step
        self.lowest_decay = lowest_decay

    def __call__(self, epoch):
        lr = max(self.lr_decay ** (epoch / self.decay_step), self.lowest_decay)
        return lr


def build_lambda_scheduler(optimizer, lr_decay=0.76, decay_step=None, lowest_decay=0.02):
    if decay_step is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, LRPolicy(lr_decay, decay_step, lowest_decay))

        return scheduler
    else:
        raise NotImplementedError()