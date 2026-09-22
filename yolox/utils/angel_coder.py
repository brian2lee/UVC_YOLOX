import code
import math

import torch
from torch import Tensor
import math
import torch.nn.functional as F
import torch.nn as nn


def smooth_l1_loss(pred, target, beta=1.0 / 9.0):
    """Smooth L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss

class CSLCoder:
    """Circular Smooth Label Coder.

    `Circular Smooth Label (CSL)
    <https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40>`_ .

    Args:
        angle_version (str): Angle definition.
        omega (float, optional): Angle discretization granularity.
            Default: 1.
        window (str, optional): Window function. Default: gaussian.
        radius (int/float): window radius, int type for
            ['triangle', 'rect', 'pulse'], float type for
            ['gaussian']. Default: 6.
    """

    def __init__(self, N=180, window='gaussian', radius=6):
        super().__init__()
        assert window in ['gaussian', 'triangle', 'rect', 'pulse']
        self.angle_range = 360
        self.angle_offset = 0
        self.window = window
        self.radius = radius
        self.encode_size = N
        self.omega = int(self.angle_range // self.encode_size)

    def encode(self, angle_targets: Tensor) -> Tensor:
        """Circular Smooth Label Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level
                Has shape (num_anchors * H * W, 1)

        Returns:
            Tensor: The csl encoding of angle offset for each scale
            level. Has shape (num_anchors * H * W, encode_size)
        """

        # radius to degree
        angle_targets_deg = angle_targets * (180 / math.pi)
        # empty label
        smooth_label = torch.zeros_like(angle_targets).repeat(
            1, self.encode_size)
        angle_targets_deg = (angle_targets_deg +
                             self.angle_offset) / self.omega
        # Float to Int
        angle_targets_long = angle_targets_deg.long()

        if self.window == 'pulse':
            radius_range = angle_targets_long % self.encode_size
            smooth_value = 1.0
        elif self.window == 'rect':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = 1.0
        elif self.window == 'triangle':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = 1.0 - torch.abs(
                (1 / self.radius) * base_radius_range)

        elif self.window == 'gaussian':
            base_radius_range = torch.arange(
                -self.angle_range // 2,
                self.angle_range // 2,
                device=angle_targets_long.device)

            radius_range = (base_radius_range +
                            angle_targets_long) % self.encode_size
            smooth_value = torch.exp(-torch.pow(base_radius_range.float(), 2.) / (2 * self.radius**2))

        else:
            raise NotImplementedError

        if isinstance(smooth_value, torch.Tensor):
            smooth_value = smooth_value.unsqueeze(0).repeat(
                smooth_label.size(0), 1)
            
        return smooth_label.scatter(1, radius_range, smooth_value)

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Circular Smooth Label Decoder.

        Args:
            angle_preds (Tensor): The csl encoding of angle offset for each
                scale level. Has shape (num_anchors * H * W, encode_size) or
                (B, num_anchors * H * W, encode_size)
            keepdim (bool): Whether the output tensor has dim retained or not.


        Returns:
            Tensor: Angle offset for each scale level. When keepdim is true,
            return (num_anchors * H * W, 1) or (B, num_anchors * H * W, 1),
            otherwise (num_anchors * H * W,) or (B, num_anchors * H * W)
        """
        if angle_preds.shape[0] == 0:
            shape = list(angle_preds.size())
            if keepdim:
                shape[-1] = 1
            else:
                shape = shape[:-1]
            return angle_preds.new_zeros(shape)
        angle_cls_inds = torch.argmax(angle_preds, dim=-1, keepdim=keepdim)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range - self.angle_offset
        return (angle_pred * (math.pi / 180)).unsqueeze(-1)

    def loss(self, pred, target, avg_factor):
        from yolox.models import gaussian_focal_loss
        loss = gaussian_focal_loss(pred, target)
        return loss.sum() / avg_factor


class VECCorder():
    def __init__(self,N):
        assert N ==2 ,"please check corder_size!!"
    def encode(self,angle_targets):
        dx = torch.sin(angle_targets[...,0]).unsqueeze(1)
        dy = torch.cos(angle_targets[...,0]).unsqueeze(1)
        return torch.cat((dx,dy),dim=-1)
    def decode(self,preds):
        vx = preds[...,0]
        vy = preds[...,1]
        vx = torch.where(torch.abs(vx)<1e-3,torch.full_like(vx,1e-3),vx)
        vy = torch.where(torch.abs(vy)<1e-3,torch.full_like(vx,1e-3),vy)
        # L = torch.sqrt(torch.pow(vx,2)+torch.pow(vy,2))
        # t = torch.acos(vy/L)
        # mask = vx<0.0
        # t =torch.where(mask,(-t%(2*math.pi)),t).unsqueeze(-1)
        mask =vy <0.0
        t = -torch.arctan((vx/vy))
        t=(
        (math.pi*mask.float()-t)%(2*math.pi)
        ).unsqueeze(-1)
        return t
    def loss(self,preds,targets,num_fgs):
        cos_d =nn.CosineSimilarity(dim=1,eps=1e-6)
        loss_c =1-cos_d(preds,targets)
        loss_c_P = torch.pow(loss_c,2)
        diff = torch.abs(preds - targets)
        # loss_sl1 = diff*loss_c.unsqueeze(1)
        # loss_sl1 = smooth_l1_loss(preds,targets)
        loss_sl1 = torch.where(diff <= 0.5, loss_c_P.unsqueeze(1)*diff,
                       1.5*loss_c.unsqueeze(1)*diff)
        return loss_sl1.sum()/num_fgs

class PSCCoder:
    """Phase-Shifting Coder.

    `Phase-Shifting Coder (PSC)
    <https://arxiv.org/abs/2211.06368>`.

    Args:
        angle_version (str): Angle definition.
            Only 'le90' is supported at present.
        dual_freq (bool, optional): Use dual frequency. Default: True.
        N (int, optional): Number of phase steps. Default: 3.
        thr_mod (float): Threshold of modulation. Default: 0.47.
    """

    def __init__(self,
                 angle_version: str = 'le90',
                 dual_freq: bool = False,
                 N: int = 3,
                 thr_mod: float = 0.47):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['le90']
        self.dual_freq = dual_freq
        self.num_step = N
        self.thr_mod = thr_mod
        if self.dual_freq:
            self.encode_size = 2 * self.num_step
        else:
            self.encode_size = self.num_step

        self.coef_sin = torch.tensor(
            tuple(
                torch.sin(torch.tensor(2 * k * math.pi / self.num_step))
                for k in range(self.num_step)))
        self.coef_cos = torch.tensor(
            tuple(
                torch.cos(torch.tensor(2 * k * math.pi / self.num_step))
                for k in range(self.num_step)))

    def encode(self, angle_targets: Tensor) -> Tensor:
        """Phase-Shifting Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1)

        Returns:
            list[Tensor]: The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
        """
        phase_targets = angle_targets 
        phase_shift_targets = tuple(
            torch.cos(phase_targets + 2 * math.pi * x / self.num_step)
            for x in range(self.num_step))

        # Dual-freq PSC for square-like problem
        if self.dual_freq:
            phase_targets = angle_targets * 4
            phase_shift_targets += tuple(
                torch.cos(phase_targets + 2 * math.pi * x / self.num_step)
                for x in range(self.num_step))

        return torch.cat(phase_shift_targets, axis=-1)

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Phase-Shifting Decoder.

        Args:
            angle_preds (Tensor): The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
            keepdim (bool): Whether the output tensor has dim retained or not.

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1) when keepdim is true,
                (num_anchors * H * W) otherwise
        """
        self.coef_sin = self.coef_sin.to(angle_preds)
        self.coef_cos = self.coef_cos.to(angle_preds)

        phase_sin = torch.sum(
            angle_preds[..., 0:self.num_step] * self.coef_sin,
            dim=-1,
            keepdim=keepdim)
        phase_cos = torch.sum(
            angle_preds[..., 0:self.num_step] * self.coef_cos,
            dim=-1,
            keepdim=keepdim)
        phase_mod = phase_cos**2 + phase_sin**2
        phase = -torch.atan2(phase_sin, phase_cos)  # In range [-pi,pi)

        if self.dual_freq:
            phase_sin = torch.sum(
                angle_preds[..., self.num_step:(2 * self.num_step)] *
                self.coef_sin,
                dim=-1,
                keepdim=keepdim)
            phase_cos = torch.sum(
                angle_preds[..., self.num_step:(2 * self.num_step)] *
                self.coef_cos,
                dim=-1,
                keepdim=keepdim)
            phase_mod = phase_cos**2 + phase_sin**2
            phase2 = -torch.atan2(phase_sin, phase_cos) / 2

            # Phase unwarpping, dual freq mixing
            # Angle between phase and phase2 is obtuse angle
            idx = torch.cos(phase) * torch.cos(phase2) + torch.sin(
                phase) * torch.sin(phase2) < 0
            # Add pi to phase2 and keep it in range [-pi,pi)
            phase2[idx] = phase2[idx] % (2 * math.pi) - math.pi
            phase = phase2

        # Set the angle of isotropic objects to zero
        phase[phase_mod < self.thr_mod] *= 0
        angle_pred = phase
        return angle_pred.unsqueeze(-1)

    def loss(self, pred, target, avg_factor):
        loss = smooth_l1_loss(pred, target)
        # loss = 
        return loss.sum() / avg_factor / self.encode_size

class PseudoAngleCoder:
    """Pseudo Angle Coder."""

    def __init__(self, N=-1):
        self.encode_size = 1

    def encode(self, angle_targets: Tensor) -> Tensor:
        return angle_targets

    def decode(self, angle_preds: Tensor) -> Tensor:

        return angle_preds
        
    def loss(self, pred, target, avg_factor):
        loss = smooth_l1_loss(pred, target)
        # cos_d =nn.CosineSimilarity(dim=1,eps=1e-6)
        # loss_c =1-cos_d(pred,target)
        # beta=1.0 / 9.0
        # diff = torch.abs(pred - target)
        # loss_c = 1-torch.cos(diff)
        # # loss_c = torch.pow(loss_c,2)
        # loss_sl1 = torch.where(diff < beta, 0.5 * diff * diff / beta,
        #                diff - 0.5 * beta)
        # loss_sl1 = torch.where(diff <= beta, loss_c*diff,
        #                diff)

        return loss.sum() / avg_factor


