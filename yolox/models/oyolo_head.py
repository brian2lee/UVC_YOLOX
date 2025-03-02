#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from cmath import cos
import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import  cxcywh2xyxy, meshgrid, visualize_assign ,cal_iou ,box2corners_th,poly2hdd,theta2vector


from .losses import IOUloss ,PIOUloss, KFLoss 
from .network_blocks import BaseConv, DWConv


class OritateHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        coder_type = "vec",
        coder_N = 2,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()
        self.training=False
        self.num_classes = num_classes
        self.coder_type = coder_type
        self.coder_size = coder_N
        self.decode_in_inference = True  # for deploy, set to False
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.ang_convs = nn.ModuleList()
        self.angle_preds=nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.ang_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act= act,
                        ),
                    ]
                )
            )

            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            #angle parameter
            self.angle_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.coder_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    
                )
            )


        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none",loss_type="giou")
        self.piou_loss =PIOUloss(reduction="none")
        self.kfiou_loss = KFLoss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
        if self.coder_type =="vec":
            from yolox.utils.angel_coder import VECCorder
            self.coder = VECCorder(N=self.coder_size)
        elif self.coder_type == "csl":
            from yolox.utils.angel_coder import CSLCoder
            self.coder = CSLCoder(N=self.coder_size)
        elif self.coder_type == "psc":
            from yolox.utils.angel_coder import PSCCoder
            self.coder = PSCCoder(N=self.coder_size)
        elif self.coder_type =="ang":
            from yolox.utils.angel_coder import PseudoAngleCoder
            self.coder = PseudoAngleCoder(N=self.coder_size)            
    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.angle_preds:
            b=conv.bias.view(1,-1)
            b.data.fill_(-math.log(1))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv,ang_conv,stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs,self.ang_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            ang_x = x
            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            ang_feat = ang_conv(ang_x)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            ang_output=self.angle_preds[k](ang_feat)
            if self.coder_type == "csl"  :
                ang_output =nn.Sigmoid()(ang_output)
            elif self.coder_type == "psc" :
                ang_output =nn.Tanh()(ang_output)
            # ang_output =nn.Tanh()(ang_output)
            # ang_output =nn.ReLU6()(ang_output)
            # ang_output =(ang_output/3)-1

            if self.training:
                output = torch.cat([reg_output,ang_output ,obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, 1, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    ang_output = ang_output.view(
                        batch_size, 1, self.coder_size, hsize, wsize
                    )
                    ang_output = ang_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, self.coder_size
                    )
                    rreg_output =torch.cat([reg_output,ang_output],dim=-1)
                    origin_preds.append(rreg_output.clone())
                
            else:
               
                # ang_output[:,2:,:,:]=(ang_output[:,2:,:,:].sigmoid())
                # output = torch.cat(
                #     [reg_output,(ang_output.sigmoid())*2*math.pi, obj_output.sigmoid(), cls_output.sigmoid()], 1
                # )
                output = torch.cat(
                    [reg_output,ang_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)

            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:

                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.coder_size + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        a= self.coder.decode(outputs[... ,4:4+self.coder_size])
        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            self.coder.decode(outputs[... ,4:4+self.coder_size]),
            outputs[..., 4+self.coder_size:]
        ], dim=-1)
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]3.0

        '''
        (x,y,w,h,dx,dy)
        '''
        ang_preds=outputs[:,:,4:4+self.coder_size]
        obj_preds = outputs[:, :, 4+self.coder_size:5+self.coder_size]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5+self.coder_size:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        ang_targets=[]
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        


        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 5))
                '''
                    (x,y,w,h,dx,dy)
                '''
                ang_target = outputs.new_zeros((0, self.coder_size))
                '''
                    (x,y,w,h,t)
                '''
                # ang_target = outputs.new_zeros((0, 1))

                l1_target = outputs.new_zeros((0, 6))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
     
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:6]
                '''
                    (x,y,w,h,dx,dy)
                '''
                gt_angles =self.coder.encode(labels[batch_idx,:num_gt,5].unsqueeze(1))
                # gt_angles= theta2vector(labels[batch_idx,:num_gt,5].unsqueeze(1))
                
                '''
                    (x,y,w,h,t)
                '''
                # gt_angles = labels[batch_idx,:num_gt,5]
                gt_classes = labels[batch_idx, :num_gt,0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_angles,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        ang_preds,
                        cls_preds,
                        obj_preds,
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_angles,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        ang_preds,
                        cls_preds,
                        obj_preds,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                ang_target = gt_angles[matched_gt_inds]
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                # rreg_target = torch.cat((reg_target,ang_target[:,:2]),dim=1)

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 6)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            ang_targets.append(ang_target)
            # rreg_targets.append(rreg_target)
            
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        ang_targets=torch.cat(ang_targets,0)
        reg_targets = torch.cat(reg_targets, 0)
        # rreg_targets = torch.cat(rreg_targets,0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        # a= torch.cat((bbox_preds,ang_preds),dim=2)
        '''
        IoU loss
        '''
        loss_iou =(
            self.piou_loss(torch.cat((bbox_preds,self.coder.decode(ang_preds)),dim=2).view(-1, 5)[fg_masks], reg_targets)
        ).sum() /num_fg

        # loss_iou =(
        #     self.kfiou_loss(torch.cat((bbox_preds,self.coder.decode(ang_preds)),dim=2).view(-1, 5)[fg_masks], reg_targets)
        # ).sum() /num_fg

        # bbox_h =box2corners_th(torch.cat((bbox_preds,vector2theta(ang_preds)),dim=2).view(-1, 5)[fg_masks])
        # bbox_h = poly2hdd(bbox_h)
        # target_h = box2corners_th(reg_targets)
        # target_h = poly2hdd(target_h)
        # loss_iou = (
        #     self.iou_loss(bbox_h, target_h)
        # ).sum() / num_fg
        '''
        class & foreground loss
        '''
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg

        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg


        vectors_preds = ang_preds.view(-1,self.coder_size)[fg_masks]
        gt_vectors = ang_targets
        """
        vector loss
        """
        # cos_d =nn.CosineSimilarity(dim=1,eps=1e-6)

        # loss_c =1-cos_d(vectors_preds,gt_vectors)
        # # loss_c= torch.nan_to_num(loss_c,2.0)
        # loss_c = torch.pow(loss_c,2)
        # # loss_R =(loss_c*delta_ang).sum()/num_fg
        # # loss_c = (delta_ang*loss_c).sum()/num_fg
        """
        delta theta multple cos loss
        """
        # loss_ang_ = (
        #     torch.pow(ang_preds[:,:,0:1].view(-1,1)[fg_masks]-ang_targets[:,0:1],2)
        # ).sum() / num_fg
        # ang_preds = ang_preds.sigmoid()
        # ang_preds = ang_preds*2*math.pi
        # ang_preds=torch.cat((torch.cos(ang_preds),torch.sin(ang_preds)),dim=-1)
        # ang_preds = ang_preds %math.pi
        # delta_ang = torch.abs(ang_preds[:,:,0].view(-1,1)[fg_masks]-ang_targets.unsqueeze(1))
        # loss_s = torch.sin(delta_ang)
        # loss_c = torch.abs(1-torch.cos(delta_ang))
        # loss_c = 1-torch.cos(ang_preds[:,:,0].view(-1,1)[fg_masks]-ang_targets.unsqueeze(1))

        # loss_R = (loss_c*delta_ang).sum()/num_fg
        # loss_R=(
        #     torch.pow(delta_ang,2)
        # ).sum() / num_fg
        """
        vector smooth_L1_loss
        # """
        # beta=1.0 / 6.0
        # diff = torch.abs(vectors_preds - gt_vectors)
        # # loss_sl1 = torch.where(diff < beta, 0.5 * diff * diff / beta,
        # #                diff - 0.5 * beta)
        # loss_sl1 = torch.where(diff <= beta, loss_c.unsqueeze(1)*diff,
        #                diff)
        # loss_sl1 = (loss_sl1).sum()/num_fg
        # loss_R = torch.clamp(loss_R,min = 1e-16,max= 18.99)
        # loss_R = loss_c*loss_sl1
        loss_ang = self.coder.loss(vectors_preds,gt_vectors,num_fg)

        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds[...,0:4].view(-1, 4)[fg_masks], l1_targets[...,0:4])
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight=3.0
        loss_ang= 5.0*loss_ang
        # #for experiment
        # loss_ang =0.0
        # loss_l1 =0.0
        loss = reg_weight * loss_iou +loss_obj +loss_cls + loss_l1 +loss_ang
        # if loss >50 or loss<0:
        #     print(loss)
        # loss=reg_weight * loss_iou + loss_obj + loss_cls + loss_l1 
        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_ang,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        l1_target[:,4:6] =theta2vector(gt[:,4].unsqueeze(1)) 
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_angles,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        angl_preds,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):

        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )
        '''
        (x,y,w,h,dx,dy)
        '''
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        angl_preds_v = angl_preds[batch_idx][fg_mask]
        
        ang_preds_theta = self.coder.decode(angl_preds_v)
        ang_preds_theta=torch.where(torch.isnan(ang_preds_theta), torch.full_like(ang_preds_theta, 0), ang_preds_theta)
        # angl_preds_=angl_preds[batch_idx][fg_mask]
        rbbox_preds = torch.cat((bboxes_preds_per_image,ang_preds_theta),dim=1)
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

    

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        
        # rbbox_preds = rbbox_preds.unsqueeze(0)
        
        # gt_rbbox = torch.cat((gt_bboxes_per_image,gt_angles[:,0:2]),dim=1).unsqueeze(0)
        '''
        iou
        '''
        # bbox_h =box2corners_th(rbbox_preds)
        # bbox_h = poly2hdd(bbox_h)
        # target_h = box2corners_th(gt_bboxes_per_image)
        # target_h = poly2hdd(target_h)
        # pair_wise_ious = bboxes_iou(target_h, bbox_h, True)
        # print("pred:")
        # print(rbbox_preds)
        '''
        Piou
        '''
        pair_wise_ious= cal_iou(gt_bboxes_per_image,rbbox_preds)
        pair_wise_ious=torch.where(torch.isnan(pair_wise_ious), torch.full_like(pair_wise_ious, 0), pair_wise_ious)
        # pair_wise_ious = pair_wise_ious.squeeze(0)
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()

            angl_preds_v=(
                angl_preds_v.float())
            angl_preds_=angl_preds_v.unsqueeze(0).repeat(num_gt,1, 1)
            gt_angles=gt_angles.unsqueeze(1).repeat(1,num_in_boxes_anchor, 1)

            # cos_d =nn.CosineSimilarity(dim=2,eps=1e-6)
            # pair_wise_ang_loss =torch.pow(1-cos_d(angl_preds_,gt_angles),2)*0.25
            # pair_wise_ang_loss=-torch.log(pair_wise_ang_loss + 1e-8)


            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_,angl_preds_


        cost = (
            pair_wise_cls_loss
            + 5.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask,self.use_l1)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        if gt_bboxes_per_image.shape[-1]==5:
            gt_bboxes_per_image = box2corners_th(gt_bboxes_per_image)
            gt_bboxes_per_image = poly2hdd(gt_bboxes_per_image)
            xy = (gt_bboxes_per_image[:,0:2] + gt_bboxes_per_image[:,2:4])/2
            wh = gt_bboxes_per_image[:,2:4] - gt_bboxes_per_image[:,0:2]

            gt_bboxes_per_image = torch.cat((xy,wh),dim= 1) 

        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center  
        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask,l1):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        if torch.any(pair_wise_ious < 0) or torch.any(pair_wise_ious > 1):
            print("Input values are out of range!")
        # if torch.isnan(pair_wise_ious).any() or (pair_wise_ious<0).any():
        #     print(torch.where(torch.isnan(pair_wise_ious)))
        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)

        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):

            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def visualize_assign_result(self, xin, labels=None, imgs=None, save_prefix="assign_vis_"):
        # original forward logic
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
        # TODO: use forward logic here.

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.full((1, grid.shape[1]), stride_this_level).type_as(xin[0])
            )
            outputs.append(output)

        outputs = torch.cat(outputs, 1)
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        for batch_idx, (img, num_gt, label) in enumerate(zip(imgs, nlabel, labels)):
            img = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8)
            num_gt = int(num_gt)
            if num_gt == 0:
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = label[:num_gt, 1:5]
                gt_classes = label[:num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                _, fg_mask, _, matched_gt_inds, _ = self.get_assignments(  # noqa
                    batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image, expanded_strides, x_shifts,
                    y_shifts, cls_preds, obj_preds,
                )

            img = img.cpu().numpy().copy()  # copy is crucial here
            coords = torch.stack([
                ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
                ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
            ], 1)

            xyxy_boxes = cxcywh2xyxy(gt_bboxes_per_image)
            save_name = save_prefix + str(batch_idx) + ".png"
            img = visualize_assign(img, xyxy_boxes, coords, matched_gt_inds, save_name)
            logger.info(f"save img to {save_name}")

if __name__ =="__main__":
    input=[torch.randn([2,256,80,80]),torch.randn([2,512,40,40]),torch.randn([2,1024,20,20])]
    out= OritateHead(80)(input)
    for item in out:
        print(item.shape)