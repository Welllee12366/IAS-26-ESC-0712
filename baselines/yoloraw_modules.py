# -*- coding: utf-8 -*-
"""
Custom layers for the YOLO-RAW reproduction (Munir et al., IEEE T-ITS 2025).

The authors have not released code (the official repository only states that the
code "will be published, once paper will be published"), so the two novel blocks
that the paper *names* are re-implemented here from their published descriptions:

  * SPP-Extended  -- Sec. III-A1, Fig. 3(a)
      The stock YOLOv5 SPPF runs three 5x5 max-pool layers in series, which is
      equivalent to 5x5 / 9x9 / 13x13 kernels. SPP-Extended adds a *fourth*
      branch, making the largest effective kernel 17x17, and concatenates the
      four branches with the un-pooled input before the 1x1 fusion conv.

  * SimAM         -- Sec. III-B1, Eqs. (1)-(2), Fig. 6 (Yang et al., ICML'21)
      A parameter-free 3-D attention module. The energy function below is the
      closed-form minimum of Eq. (2) of the paper; a lower energy means the
      neuron is better separated from its neighbours and therefore receives a
      larger attention weight.

Both modules are channel-preserving, which is why parse_model's default
`c2 = ch[f]` branch is the correct bookkeeping for them.

NOTE on fidelity (see train_scripts/README.md, "YOLO-RAW reproduction"):
Ultralytics 8.x has no anchor-based YOLOv5 detection head -- `Detect` is the
DFL / anchor-free head even when `legacy=True`. The paper's YOLO-RAW is an
anchor-based YOLOv5(YOLOv5m). The four architectural contributions of YOLO-RAW
are reproduced exactly; the anchor assignment/regression mechanism follows the
modern Ultralytics head, which is the same head every other row of our Table IV
is trained with. This is stated as prose in the manuscript.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


def _make_divisible(x, divisor=8):
    """Match the channel rounding used inside ultralytics.nn.tasks.parse_model."""
    return math.ceil(x / divisor) * divisor


class SPPE(nn.Module):
    """SPP-Extended: YOLOv5 SPPF with an extra (4th) 5x5 max-pool branch.

    Effective receptive kernels: 5x5, 9x9, 13x13, 17x17 (the paper's 17x17
    branch, Sec. III-A1). Channels are preserved, as in SPPF.

    Args:
        c1 (int): input channels
        c2 (int): output channels (== the width-scaled c1 in our YAML)
        k  (int): max-pool kernel, 5 in the paper
        n  (int): number of sequential max-pool branches (4 for SPP-Extended)
    """

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 4):
        super().__init__()
        c_ = c1 // 2  # hidden channels, same 0.5 expansion as SPPF
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        return self.cv2(torch.cat(y, 1))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class SimAM(nn.Module):
    """SimAM: parameter-free 3-D attention (Yang et al., ICML 2021), Eq. (2).

    e* = 4 (sigma^2 + lambda) / ((t - mu)^2 + 2 sigma^2 + 2 lambda)
    Lower energy -> better linear separability -> stronger attention weight.

    Deliberately defines **no** trainable parameters, which is the whole point of
    the module in the paper (Sec. III-B1: "without adding any parameters to our
    original model"). `c1`/`c2` are accepted only so the block can be written in
    the standard YAML style.
    """

    def __init__(self, c1=None, c2=None, e_lambda: float = 1e-4):
        super().__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w - 1
        d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)  # (t - mu)^2
        v = d.sum(dim=(2, 3), keepdim=True) / n            # sigma^2
        # + 0.5 is a monotone shift so that the sigmoid centre is a fair prior
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return x * self.act(e_inv)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(e_lambda={self.e_lambda})"


def register() -> None:
    """Make SimAM / SPPE resolvable by ultralytics.nn.tasks.parse_model.

    parse_model looks the module name up in its own module globals, so injecting
    the classes there is sufficient (and is the standard way to add custom
    blocks without forking ultralytics).
    """
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    for mod in (tasks, modules):
        mod.SimAM = SimAM
        mod.SPPE = SPPE
    # also expose the factory so YAMLs can be written with the plain class names
    tasks.yoloraw_modules = (SimAM, SPPE)
