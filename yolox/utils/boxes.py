#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import numpy as np
import math
import torch
import torchvision
from cuda_op.cuda_ext import sort_v
from .rot_utils import box2corners_th
EPSILON = 1e-8

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cxcywh2xyxy"
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]
def del_diamond(pred):

    v1=(pred[:,3]-pred[:,1])/((1-pred[:,4])*(pred[:,2]-pred[:,0]))
    v2=(pred[:,4]*(pred[:,2]-pred[:,0]))/(pred[:,5]*(pred[:,3]-pred[:,1]))
    dv=v1-v2
    mask=dv>=-0.5
    mask=mask.unsqueeze(1)
    mask = torch.mul(mask.cpu(), torch.ones(1, 22).cpu())
    pred=torch.masked_select(pred,mask.bool())
    pred= pred.reshape(-1, 22)
    return pred

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    num_box_para=prediction.shape[2]-num_classes
    if num_box_para ==6:
        pass


    else:
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
    
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        # if num_box_para==9:
        #     image_pred=del_diamond(image_pred)
            
        class_conf, class_pred = torch.max(image_pred[:, num_box_para:  num_box_para+ num_classes], 1, keepdim=True)


        conf_mask = (image_pred[:, num_box_para-1] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :num_box_para], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            if  num_box_para ==6:
                box_corner=box2corners_th(detections)
                box_corner =box_corner.view(-1,8)
                box_corner_x = box_corner[:,0::2]
                box_corner_y = box_corner[:,1::2]
                hbb_box =torch.cat((torch.min(box_corner_x,dim =1)[0].unsqueeze(1),torch.min(box_corner_y,dim =1)[0].unsqueeze(1)
                            ,torch.max(box_corner_x,dim =1)[0].unsqueeze(1),torch.max(box_corner_y,dim =1)[0].unsqueeze(1)),dim=1)
                nms_out_index = torchvision.ops.nms(
                    hbb_box[:, :4],
                    detections[:, num_box_para-1] * detections[:, num_box_para],
                    nms_thre,
                )    
            else:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, num_box_para-1] * detections[:, num_box_para],
                    nms_thre,
                )
        else:
            if  num_box_para ==6:
                box_corner=box2corners_th(detections)
                box_corner =box_corner.view(-1,8)
                box_corner_x = box_corner[:,0::2]
                box_corner_y = box_corner[:,1::2]
                hbb_box =torch.cat((torch.min(box_corner_x,dim =1)[0].unsqueeze(1),torch.min(box_corner_y,dim =1)[0].unsqueeze(1)
                            ,torch.max(box_corner_x,dim =1)[0].unsqueeze(1),torch.max(box_corner_y,dim =1)[0].unsqueeze(1)),dim=1)
                nms_out_index = torchvision.ops.nms(
                    hbb_box[:, :4],
                    detections[:, num_box_para-1] * detections[:, num_box_para],
                    nms_thre,
                )    
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, num_box_para-1] * detections[:, num_box_para],
                    detections[:, num_box_para+1],
                    nms_thre,
                )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def cxcywh2xyxy(bboxes):
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return bboxes




        
