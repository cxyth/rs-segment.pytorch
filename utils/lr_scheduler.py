# https://blog.csdn.net/qq_31580989/article/details/121491181
# https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
import warnings
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class PolyScheduler(_LRScheduler):
    r"""
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.PolyScheduler(optimizer, min_lr=0.01, steps_per_epoch=None, epochs=10)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>     scheduler.step()


    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """

    def __init__(self,
                 optimizer,
                 power=1.0,
                 total_steps=None,
                 epochs=None,
                 steps_per_epoch=None,
                 min_lr=0,
                 last_epoch=-1,
                 verbose=False):

        # Validate optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        # self.by_epoch = by_epoch
        self.epochs = epochs
        self.min_lr = min_lr
        self.power = power

        # check param
        param_dic = {'total_steps': total_steps, 'epochs': epochs, 'steps_per_epoch': steps_per_epoch}
        for k, v in param_dic.items():
            if v is not None:
                if v <= 0 or not isinstance(v, int):
                    raise ValueError("Expected positive integer {}, but got {}".format(k, v))

        # Validate total_steps
        if total_steps is not None:
            self.total_steps = total_steps
        elif epochs is not None and steps_per_epoch is None:
            self.total_steps = epochs
        elif epochs is not None and steps_per_epoch is not None:
            self.total_steps = epochs * steps_per_epoch
        else:
            raise ValueError("You must define either total_steps OR epochs OR (epochs AND steps_per_epoch)")

        super(PolyScheduler, self).__init__(optimizer, last_epoch, verbose)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                             .format(step_num + 1, self.total_steps))

        coeff = (1 - step_num / self.total_steps) ** self.power

        return [(base_lr - self.min_lr) * coeff + self.min_lr for base_lr in self.base_lrs]
