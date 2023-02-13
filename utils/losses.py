# https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/joint_loss.py
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

__all__ = ["JointLoss", "WeightedLoss", "MSELoss", "SoftBCELoss"]


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)


class MSELoss(nn.Module):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
    """
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor, mask: Tensor = None) -> Tensor:

        loss = F.mse_loss(y_pred, y_true, reduction="none")

        if mask is not None:
            loss *= mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


class RMSELoss(nn.Module):
    r"""Creates a criterion that measures the root mean square error (RMSE) between
    each element in the input :math:`x` and target :math:`y`.

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
    """
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean', eps=1e-6) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred: Tensor, y_true: Tensor, mask: Tensor = None) -> Tensor:

        loss = torch.sqrt(F.mse_loss(y_pred, y_true, reduction="none") + self.eps)

        if mask is not None:
            loss *= mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


class SoftBCELoss(nn.Module):

    __constants__ = [
        "weight",
        "reduction",
        "ignore_index",
        "smooth_factor",
    ]

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = -100,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
    ):
        """Drop-in replacement for torch.nn.BCELoss with few additions: ignore_index and label_smoothing

        Args:
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])

        Shape
             - **y_pred** - torch.Tensor of shape NxCxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)

        Returns:
            loss: torch.Tensor
        """

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)
        else:
            soft_targets = y_true

        loss = F.binary_cross_entropy(y_pred, soft_targets, weight=self.weight, reduction="none")

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss
