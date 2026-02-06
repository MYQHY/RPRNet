import math
from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _triple(v) -> Tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    assert len(v) == 3
    return tuple(int(x) for x in v)


class DeformConv3d(nn.Module):
    """
    Deformable 3D convolution (reference implementation with grid_sample),
    with explicit bounds on spatial (x,y) and temporal/depth (z) offsets.

    Input:  x [B, Cin, D, H, W]
    Output: y [B, Cout, D_out, H_out, W_out]

    Key feature:
      - max_disp_xy: maximum absolute displacement in x/y (pixels)
      - max_disp_t : maximum absolute displacement in z (depth/time index)

    Offsets are predicted and then bounded as:
      dx = tanh(raw_dx) * max_disp_xy
      dy = tanh(raw_dy) * max_disp_xy
      dz = tanh(raw_dz) * max_disp_t
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        # Bounds (in voxel units)
        max_disp_xy: float = 0.0,
        max_disp_t: float = 0.0,
        # Offset conv hyperparams (optional)
        offset_kernel_size: Optional[Union[int, Tuple[int, int, int]]] = None,
        offset_padding: Optional[Union[int, Tuple[int, int, int]]] = None,
    ):
        super().__init__()
        kD, kH, kW = _triple(kernel_size)
        sD, sH, sW = _triple(stride)
        pD, pH, pW = _triple(padding)
        dD, dH, dW = _triple(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kD, kH, kW)
        self.stride = (sD, sH, sW)
        self.padding = (pD, pH, pW)
        self.dilation = (dD, dH, dW)

        # Bounds
        if max_disp_xy < 0 or max_disp_t < 0:
            raise ValueError("max_disp_xy and max_disp_t must be >= 0")
        self.max_disp_xy = float(max_disp_xy)
        self.max_disp_t = float(max_disp_t)

        self.K = kD * kH * kW

        # Weight parameter for sampled K points
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, self.K))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        # Offset predictor: outputs raw offsets for (dx,dy,dz) per kernel point
        ok = offset_kernel_size if offset_kernel_size is not None else 3
        op = offset_padding if offset_padding is not None else 1
        okD, okH, okW = _triple(ok)
        opD, opH, opW = _triple(op)

        self.offset_conv = nn.Conv3d(
            in_channels,
            3 * self.K,
            kernel_size=(okD, okH, okW),
            stride=(sD, sH, sW),
            padding=(opD, opH, opW),
            bias=True,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.K
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Important: start with zero offsets
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

    @torch.no_grad()
    def _make_kernel_relative_coords(self, device, dtype):
        kD, kH, kW = self.kernel_size
        dD, dH, dW = self.dilation

        dz = torch.arange(-(kD // 2), kD // 2 + 1, device=device, dtype=dtype) * dD
        dy = torch.arange(-(kH // 2), kH // 2 + 1, device=device, dtype=dtype) * dH
        dx = torch.arange(-(kW // 2), kW // 2 + 1, device=device, dtype=dtype) * dW

        zz, yy, xx = torch.meshgrid(dz, dy, dx, indexing="ij")
        rel = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # [K,3] (x,y,z)
        return rel

    def _bound_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        """
        offsets: [B, 3K, D_out, H_out, W_out] raw offsets
        return : [B, K, D_out, H_out, W_out, 3] bounded offsets in (x,y,z)
        """
        B, _, D_out, H_out, W_out = offsets.shape
        off = offsets.view(B, self.K, 3, D_out, H_out, W_out)  # (dx,dy,dz)
        # Apply tanh bounds
        dx = torch.tanh(off[:, :, 0, ...]) * self.max_disp_xy
        dy = torch.tanh(off[:, :, 1, ...]) * self.max_disp_xy
        dz = torch.tanh(off[:, :, 2, ...]) * self.max_disp_t

        off_b = torch.stack([dx, dy, dz], dim=-1)  # [B,K,Do,Ho,Wo,3]
        return off_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        


        if x.dim() != 5:
            raise ValueError("Expected x as [B, C, D, H, W]")

        B, C, D, H, W = x.shape
        sD, sH, sW = self.stride
        pD, pH, pW = self.padding

        # Raw offsets at output resolution
        raw_offsets = self.offset_conv(x)  # [B, 3K, D_out, H_out, W_out]
        _, _, D_out, H_out, W_out = raw_offsets.shape

        # Bound offsets
        offsets = self._bound_offsets(raw_offsets)  # [B,K,Do,Ho,Wo,3] (x,y,z)
        self._last_offsets = offsets

        # Pad input like standard conv
        x_pad = F.pad(x, (pW, pW, pH, pH, pD, pD))
        _, _, Dp, Hp, Wp = x_pad.shape

        device = x.device
        dtype = x.dtype

        # Base sampling centers for each output voxel (in padded coords)
        oz = torch.arange(D_out, device=device, dtype=dtype)
        oy = torch.arange(H_out, device=device, dtype=dtype)
        ox = torch.arange(W_out, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(oz, oy, ox, indexing="ij")

        # base = torch.stack([xx * sW, yy * sH, zz * sD], dim=-1).unsqueeze(0)  # [1,Do,Ho,Wo,3]
        kD, kH, kW = self.kernel_size
        dD, dH, dW = self.dilation

        cx = (kW // 2) * dW
        cy = (kH // 2) * dH
        cz = (kD // 2) * dD

        base = torch.stack(
            [xx * sW + cx, yy * sH + cy, zz * sD + cz],
            dim=-1
        ).unsqueeze(0)  # [1,Do,Ho,Wo,3]
        # Kernel relative coords
        rel = self._make_kernel_relative_coords(device=device, dtype=dtype)  # [K,3]
        rel = rel.view(1, self.K, 1, 1, 1, 3)  # [1,K,1,1,1,3]
        base = base.unsqueeze(1)               # [1,1,Do,Ho,Wo,3]

        # Absolute sampling grid (padded coords)
        grid_abs = base + rel + offsets  # [B,K,Do,Ho,Wo,3] in (x,y,z)

        # Normalize to [-1,1] for grid_sample
        gx = 2.0 * (grid_abs[..., 0] / max(Wp - 1, 1)) - 1.0
        gy = 2.0 * (grid_abs[..., 1] / max(Hp - 1, 1)) - 1.0
        gz = 2.0 * (grid_abs[..., 2] / max(Dp - 1, 1)) - 1.0
        grid = torch.stack([gx, gy, gz], dim=-1)  # [B,K,Do,Ho,Wo,3]

        # Sample K times (reference implementation)
        sampled_list = []
        for k in range(self.K):
            sk = F.grid_sample(
                x_pad,
                grid[:, k, ...],       # [B,Do,Ho,Wo,3]
                mode="bilinear",       # trilinear for 5D
                padding_mode="zeros",
                align_corners=True,
            )
            sampled_list.append(sk)

        sampled = torch.stack(sampled_list, dim=2)  # [B,C,K,Do,Ho,Wo]

        # Weighted sum -> output channels
        out = torch.einsum("bckdhw,ock->bodhw", sampled, self.weight)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)

        return out
