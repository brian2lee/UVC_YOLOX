#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from  yolox.utils import poly2hdd
import torch
import torch.nn as nn
import math
from cuda_op.cuda_ext import sort_v
EPSILON = 1e-8

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        
        return loss

class PIOUloss(nn.Module):
    def __init__(self,reduction):
        super(PIOUloss, self).__init__()
        self.reduction = reduction
    def decode_box(self,boxes):
        x = boxes[:,0]
        y = boxes[:,1]
        W = torch.abs(boxes[:,2])
        H = torch.abs(boxes[:,3])
        R1 = torch.clamp(boxes[:,4],0.001,0.999)
        R2 = torch.clamp(boxes[:,5],0.001,0.999)
        w1 = torch.sqrt(torch.pow(W*R1,2)+torch.pow(H*R2,2))
        w2 = torch.sqrt(torch.pow(W*(1-R1),2)+torch.pow(H*(1-R2),2))
        maxW = torch.maximum(w1,w2)
        tmp_v = w1==maxW
        T_tmp_v = w1 != maxW
        t1 = torch.full([boxes.shape[0]],0.5*math.pi,device="cuda:0")+torch.atan(-(H*R2)/(W*R1))
        t2 = torch.atan((H*(1-R2))/(W*(1-R1)))
        t = tmp_v * t1 + T_tmp_v * t2
        out_boxes = torch.cat((x.unsqueeze(1),y.unsqueeze(1),w2.unsqueeze(1),w1.unsqueeze(1),t.unsqueeze(1)),axis=1)
    
        return out_boxes
    
    def box_intersection_th(self,corners1:torch.Tensor, corners2:torch.Tensor):
        """find intersection points of rectangles
        Convention: if two edges are collinear, there is no intersection point

        Args:
            corners1 (torch.Tensor):  N, 4, 2
            corners2 (torch.Tensor):  N, 4, 2

        Returns:
            intersectons (torch.Tensor): N, 4, 4, 2
            mask (torch.Tensor) : N, 4, 4; bool
        """
        # build edges from corners
        line1 = torch.cat([corners1, corners1[:, [1, 2, 3, 0], :]], dim=2) #  N, 4, 4: Batch, Box, edge, point
        line2 = torch.cat([corners2, corners2[:, [1, 2, 3, 0], :]], dim=2)
        # duplicate data to pair each edges from the boxes
        # ( N, 4, 4) -> ( N, 4, 4, 4) : Batch, Box, edge1, edge2, point
        line1_ext = line1.unsqueeze(2).repeat([1,1,4,1])
        line2_ext = line2.unsqueeze(1).repeat([1,4,1,1])
        x1 = line1_ext[..., 0]
        y1 = line1_ext[..., 1]
        x2 = line1_ext[..., 2]
        y2 = line1_ext[..., 3]
        x3 = line2_ext[..., 0]
        y3 = line2_ext[..., 1]
        x4 = line2_ext[..., 2]
        y4 = line2_ext[..., 3]
        # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        num = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)     
        den_t = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
        t = den_t / num
        t[num == .0] = -1.
        mask_t = (t > 0) * (t < 1)                # intersection on line segment 1
        den_u = (x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)
        u = -den_u / num
        u[num == .0] = -1.
        mask_u = (u > 0) * (u < 1)                # intersection on line segment 2
        mask = mask_t * mask_u 
        t = den_t / (num + EPSILON)                 # overwrite with EPSILON. otherwise numerically unstable
        intersections = torch.stack([x1 + t*(x2-x1), y1 + t*(y2-y1)], dim=-1)
        intersections = intersections * mask.float().unsqueeze(-1)
        return intersections, mask
    def box1_in_box2(self,corners1:torch.Tensor, corners2:torch.Tensor):
        """check if corners of box1 lie in box2
        Convention: if a corner is exactly on the edge of the other box, it's also a valid point

        Args:
            corners1 (torch.Tensor): (B, N, 4, 2)
            corners2 (torch.Tensor): (B, N, 4, 2)

        Returns:
            c1_in_2: (B, N, 4) Bool
        """
        a = corners2[ :, 0:1, :]  # (N, 1, 2)
        b = corners2[:, 1:2, :]  # (N, 1, 2)
        d = corners2[:, 3:4, :]  # (N, 1, 2)
        ab = b - a                  # (N, 1, 2)
        am = corners1 - a           # (N, 4, 2)
        ad = d - a                  # (N, 1, 2)
        p_ab = torch.sum(ab * am, dim=-1)       # (N, 4)
        norm_ab = torch.sum(ab * ab, dim=-1)    # (N, 1)
        p_ad = torch.sum(ad * am, dim=-1)       # (N, 4)
        norm_ad = torch.sum(ad * ad, dim=-1)    # (N, 1)
        # NOTE: the expression looks ugly but is stable if the two boxes are exactly the same
        # also stable with different scale of bboxes
        cond1 = (p_ab / norm_ab > - 1e-6) * (p_ab / norm_ab < 1 + 1e-6)   # (N, 4)
        cond2 = (p_ad / norm_ad > - 1e-6) * (p_ad / norm_ad < 1 + 1e-6)   # ( N, 4)
        return cond1*cond2

    def box_in_box_th(self,corners1:torch.Tensor, corners2:torch.Tensor):
        """check if corners of two boxes lie in each other

        Args:
            corners1 (torch.Tensor): (N, 4, 2)
            corners2 (torch.Tensor): (N, 4, 2)

        Returns:
            c1_in_2: (N, 4) Bool. i-th corner of box1 in box2
            c2_in_1: (N, 4) Bool. i-th corner of box2 in box1
        """
        c1_in_2 = self.box1_in_box2(corners1, corners2)
        c2_in_1 = self.box1_in_box2(corners2, corners1)
        return c1_in_2, c2_in_1
    def build_vertices(self,corners1:torch.Tensor, corners2:torch.Tensor, 
                c1_in_2:torch.Tensor, c2_in_1:torch.Tensor, 
                inters:torch.Tensor, mask_inter:torch.Tensor):
        """find vertices of intersection area

        Args:
            corners1 (torch.Tensor): (B, N, 4, 2)
            corners2 (torch.Tensor): (B, N, 4, 2)
            c1_in_2 (torch.Tensor): Bool, (B, N, 4)
            c2_in_1 (torch.Tensor): Bool, (B, N, 4)
            inters (torch.Tensor): (B, N, 4, 4, 2)
            mask_inter (torch.Tensor): (B, N, 4, 4)
        
        Returns:
            vertices (torch.Tensor): (B, N, 24, 2) vertices of intersection area. only some elements are valid
            mask (torch.Tensor): (B, N, 24) indicates valid elements in vertices
        """
        # NOTE: inter has elements equals zero and has zeros gradient (masked by multiplying with 0). 
        # can be used as trick
        vertices = torch.cat([corners1, corners2, inters.view([corners1.shape[0],-1, 2])], dim=1) # ( N, 4+4+16, 2)
        mask = torch.cat([c1_in_2, c2_in_1, mask_inter.view([corners1.shape[0],-1])], dim=1) # Bool (B, N, 4+4+16)
        return vertices, mask
    def sort_indices(self,vertices:torch.Tensor, mask:torch.Tensor):
        """[summary]

        Args:
            vertices (torch.Tensor): float (N, 24, 2)
            mask (torch.Tensor): bool (N, 24)

        Returns:
            sorted_index: bool (B, N, 9)
        
        Note:
            why 9? the polygon has maximal 8 vertices. +1 to duplicate the first element.
            the index should have following structure:
                (A, B, C, ... , A, X, X, X) 
            and X indicates the index of arbitary elements in the last 16 (intersections not corners) with 
            value 0 and mask False. (cause they have zero value and zero gradient)
        """
        num_valid = torch.sum(mask.int(), dim=1).int()      # (N)
        mean = torch.sum(vertices * mask.float().unsqueeze(-1), dim=1, keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)
        vertices_normalized = vertices - mean      # normalization makes sorting easier
        mask = mask.unsqueeze(0)
        num_valid = num_valid.unsqueeze(0)
        vertices_normalized = vertices_normalized.unsqueeze(0)
        return sort_v(vertices_normalized, mask, num_valid).long()
    def calculate_area(self,idx_sorted:torch.Tensor, vertices:torch.Tensor):
        """calculate area of intersection

        Args:
            idx_sorted (torch.Tensor): (B, N, 9)
            vertices (torch.Tensor): (B, N, 24, 2)
        
        return:
            area: (B, N), area of intersection
            selected: (B, N, 9, 2), vertices of polygon with zero padding 
        """
        
        idx_ext = idx_sorted.unsqueeze(-1).repeat([1,1,1,2])
        selected = torch.gather(vertices.unsqueeze(0), 2, idx_ext)
        total = selected[:, :, 0:-1, 0]*selected[:, :, 1:, 1] - selected[:, :, 0:-1, 1]*selected[:, :, 1:, 0]
        total = torch.sum(total, dim=2)
        area = torch.abs(total) / 2
        return area, selected

    def oriented_box_intersection_2d(self,corners1:torch.Tensor, corners2:torch.Tensor):
        """calculate intersection area of 2d rectangles 

        Args:
            corners1 (torch.Tensor): (N, 4, 2)
            corners2 (torch.Tensor): (N, 4, 2)

        Returns:
            area: (N), area of intersection
            selected: (N, 9, 2), vertices of polygon with zero padding 
        """
        
        inters, mask_inter = self.box_intersection_th(corners1, corners2)
        c12, c21 = self.box_in_box_th(corners1, corners2)
        vertices, mask = self.build_vertices(corners1, corners2, c12, c21, inters, mask_inter)
        sorted_indices = self.sort_indices(vertices, mask)
        # sorted_indices = sorted_indices.view(mask.shape[0],mask.shape[1],mask.shape[2],9)
        return self.calculate_area(sorted_indices, vertices)


    def box2corners_th(self,box:torch.Tensor)-> torch.Tensor:
        """convert box coordinate to corners

        Args:
            box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha

        Returns:
            torch.Tensor: (B, N, 4, 2) corners
        """
        B = box.size()[0]
        x = box[..., 0:1]
        y = box[..., 1:2]
        w = box[..., 2:3]
        h = box[..., 3:4]
        alpha = box[..., 4:5] # (B, N, 1)
        x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).to(box.device) # (1,1,4)
        x4 = x4 * w     # (B, N, 4)
        y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).to(box.device)
        y4 = y4 * h     # (B, N, 4)
        corners = torch.stack([x4, y4], dim=-1)     # (B, N, 4, 2)
        sin = torch.sin(alpha)
        cos = torch.cos(alpha)
        row1 = torch.cat([cos, sin], dim=-1)
        row2 = torch.cat([-sin, cos], dim=-1)       # (B, N, 2)
        rot_T = torch.stack([row1, row2], dim=-2)   # (B, N, 2, 2)
        rotated = torch.bmm(corners.view([-1,4,2]), rot_T.view([-1,2,2]))
        # rotated = rotated.view([B,-1,4,2])          # (B*N, 4, 2) -> (B, N, 4, 2)
        rotated[..., 0] += x
        rotated[..., 1] += y
        return rotated
    
    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        if torch.isnan(pred).any():
            print("input_pred:")
            print(pred[torch.where(torch.isnan(target))[0][0]])
        if torch.isnan(target).any():
            print("input_target:")
            print(target[torch.where(torch.isnan(target))[0][0]])
        if pred.shape[1]==8:
            pred =pred[:,0:6].view(-1, 6)
            target = target.view(-1,6)
            tl = torch.max(
                (pred[:, :2] - pred[:, 2:4] / 2), (target[:, :2] - target[:, 2:4] / 2)
            )
            br = torch.min(
                (pred[:, :2] + pred[:, 2:4] / 2), (target[:, :2] + target[:, 2:4] / 2)
            )

            pred =self.decode_box(pred)
            target = self.decode_box(target)
        # if torch.isnan(pred).any():
        #     print("box_pred:")
        #     print(pred[torch.where(torch.isnan(target))[0][0]])
        # if torch.isnan(target).any():
        #     print("box_target:")
        #     print(target[torch.where(torch.isnan(target))[0][0]])
        pred=torch.nan_to_num(pred,nan=0.0)
        target=torch.nan_to_num(target,nan=0.0)
        corner_gt = self.box2corners_th(target)
        corner_pred = self.box2corners_th(pred)
        if pred.shape[1]==5:
            hbb_pred=poly2hdd(corner_pred)
            hdd_gt =poly2hdd(corner_gt)
            tl = torch.max(
                (hbb_pred[:, :2]), (hdd_gt[:, :2])
            )
            br = torch.min(
                (hbb_pred[:, 2:4]), (hdd_gt[:, 2:4])
            )
        inter_area, _ = self.oriented_box_intersection_2d(corner_gt, corner_pred)
        
        en = (tl < br).type(tl.type()).prod(dim=1)
        inter_area = inter_area.squeeze(0)*en
        area1 = torch.clamp(target[:, 2] * target[:, 3],min=1e-6)
        area2 = torch.clamp(pred[:, 2] * pred[:, 3],min=1e-6)
        u = area1 + area2 - inter_area
        iou = inter_area / (u+ 1e-16)
        loss = 1 - iou ** 2
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        loss = torch.nan_to_num(loss,nan=1.0)
        if torch.isnan(loss).any():
            print(loss)
            print(pred[torch.where(torch.isnan(loss))[0][0]])
            print(target[torch.where(torch.isnan(loss))[0][0]])
        loss=torch.clamp(loss,0.0,1.0)
        return loss

class KFLoss(nn.Module):
    def __init__(self,reduction):
        super(KFLoss, self).__init__()
        self.reduction = reduction
    def xy_wh_r_2_xy_sigma(self,xywhr):
        """Convert oriented bounding box to 2-D Gaussian distribution.

        Args:
            xywhr (torch.Tensor): rbboxes with shape (N, 5).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = xywhr[..., 4]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma
    def forward(self,pred,target):
        assert pred.shape[0] == target.shape[0]
        xy_p = pred[:, :2]
        xy_t = target[:, :2]
        _, Sigma_p = self.xy_wh_r_2_xy_sigma(pred)
        _, Sigma_t = self.xy_wh_r_2_xy_sigma(target)
        # Smooth-L1 norm
        diff = torch.abs(xy_p - xy_t)
        xy_loss = torch.where(diff < (1.0/9.0), 0.5 * diff * diff / (1.0/9.0),
                            diff - 0.5 * (1.0/9.0)).sum(dim=-1)
        Vb_p = 4 * Sigma_p.det().sqrt()
        Vb_t = 4 * Sigma_t.det().sqrt()
        K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
        Sigma = Sigma_p - K.bmm(Sigma_p)
        Vb = 4 * Sigma.det().sqrt()
        Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
        KFIoU = Vb / (Vb_p + Vb_t - Vb + 1e-6)
        kf_loss = (1.0-KFIoU)
        loss = (xy_loss+kf_loss).clamp(0)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss



def gaussian_focal_loss(pred: torch.Tensor,
                        gaussian_target: torch.Tensor,
                        alpha: float = 2.0,
                        gamma: float = 4.0,
                        pos_weight: float = 1.0,
                        neg_weight: float = 1.0) -> torch.Tensor:
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    l = pos_weight * pos_loss + neg_weight * neg_loss
    return pos_weight * pos_loss + neg_weight * neg_loss


if __name__ =="__main__":
    k =KFLoss(None)
    gt = torch.tensor([[50,50,20,10,0.3*math.pi],[50,50,10,30,1.8*math.pi]])
    pred =torch.tensor([[50,50,20,10,0.3*math.pi],[50,50,10,30,0.3*math.pi]])
    l =k(pred,gt)
    print(l)