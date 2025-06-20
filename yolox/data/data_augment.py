#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

from yolox.utils import boxes, xyxy2cxcywh,obb2poly_np,aoI_select



def test_img(img,labels=None):
    cv2.namedWindow("aa",cv2.WINDOW_NORMAL)
    VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

    if labels is not None:
        c = obb2poly_np(labels)
        ch=c.reshape(len(labels),8)
        for index,i in enumerate(ch):
            corners=i
            corners_x =corners[0::2]
            corners_y =corners[1::2]
            # x1=int(i[0]-i[2]/2) 
            # y1=int(i[1]-i[3]/2)
            # x2=int(i[0]+i[2]/2)
            # y2=int(i[1]+i[3]/2)
            # x1=int(i[0]) 
            # y1=int(i[1])
            # x2=int(i[2])
            # y2=int(i[3])
            cv2.line(img,(int(corners_x[0]),int(corners_y[0])),(int(corners_x[1]),int(corners_y[1])),[0,0,255],1)
            cv2.line(img,(int(corners_x[1]),int(corners_y[2])),(int(corners_x[3]),int(corners_y[3])),[0,0,255],1)
            cv2.line(img,(int(corners_x[0]),int(corners_y[0])),(int(corners_x[3]),int(corners_y[3])),[0,0,255],1)
            
            v_y=int(0.5*labels[index][3]*math.cos(-labels[index][4]))
            v_x=int(0.5*labels[index][3]*math.sin(-labels[index][4]))

            cv2.line(img,(int(labels[index][0]),int(labels[index][1])),(int(labels[index][0]-v_x),int(labels[index][1]-v_y)),[0,0,255],1)# if x1<0:
            #     x1=0
            # if y1<0:
            #     y1=0
            # if x2>img.shape[1]:
            #     x2=img.shape[1]
            # if y2>img.shape[0]:
            #     y2=img.shape[0]
            # cv2.rectangle(img,(x1,y1),(x2,y2),[255,0,0],3)
            # cv2.putText(img,VOC_CLASSES[int(i[4])],(int(i[0]),int(i[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),1,cv2.LINE_AA)
    cv2.imshow("aa",img)
    cv2.waitKey(0)



def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    # angle = get_aug_params(degrees)
    angle = float(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )
    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes
    return targets

def apply_affine_to_bboxes_ori(targets, target_size, M, scale, degree):
    num_gts = len(targets)
    box_index=[]
    targets_o=()
    # warp corner points
    twidth, theight = target_size
    targets_c =obb2poly_np(targets)
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets_c.reshape(4 * num_gts,2)
    # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)
    for i in range(len(targets)):
        corner_points_i =corner_points[i]
        corners_x =corner_points_i[0::2]
        corners_y =corner_points_i[1::2]
        if not aoI_select([corners_x.min(),corners_y.min(),corners_x.max(),corners_y.max()],[twidth,theight],threshold=0.6):
            box_index.append(i)
        # if corners_x.min()>0 and corners_x.max()<twidth and corners_y.min()>0 and corners_y.max()<theight:
        #     box_index.append(i)
    # if not len(box_index)==0:
        # create new boxes
    xys_o = (corner_points[box_index,0:2]+corner_points[box_index,4:6])/2
    hws_o  = targets[box_index,2:4]*scale
    angs_o = targets[box_index,4:5]-math.pi*(degree/180)
    cls_o = targets[box_index,5:]
    targets_o =np.concatenate([xys_o,hws_o,angs_o,cls_o],axis=1)



    return targets_o



def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
    rotate ="hd",
    
):
    d =[0,5,10,15,18]
    target_o =[]
    while(len(target_o)==0):
        degrees = d[random.randint(0,len(d)-1)]
        M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

        img_o = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

        # Transform label coordinates
        if len(targets) > 0:
            if rotate == "ROT" or rotate == "hd":
                target_o = apply_affine_to_bboxes(targets, target_size, M, scale)
            elif rotate =="ORI":
                target_o = apply_affine_to_bboxes_ori(targets, target_size, M, scale,degrees)   
    # test_img(img_o,target_o)
    return img_o, target_o


def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def _mirror_R(image, boxes, head_type,prob=0.5 ):
    #flip 180
    height, width, _ = image.shape
    
    if random.random() < prob:
        image = image[::-1, ::-1]
        if head_type=="ROT":
            boxes[:,0] = width - boxes[:,0]
            boxes[:,1] = height - boxes[:,1]
            boxes[:,2] = width - boxes[:,2]
            boxes[:,3] = height - boxes[:,3]
            boxes[:,6:8]=abs(1-boxes[:,6:8])
        elif head_type=="ORI":
            boxes[:,0] = width - boxes[:,0]
            boxes[:,1] = height - boxes[:,1]
            boxes[:,4] = (boxes[:,4]+math.pi)%(2*math.pi)

    return image, boxes

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r,resized_img


class TrainTransform:
    def __init__(self, max_labels=None, flip_prob=0.5, hsv_prob=1.0,head_type=None):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.head_type = head_type


    def __call__(self, image, targets, input_dim):

        if self.head_type=="ROT":
            boxes = targets[:, :8].copy()
            labels = targets[:, 8].copy()
        elif self.head_type=="hd":    
            boxes = targets[:, :4].copy()
            labels = targets[:, 4].copy()
        elif self.head_type == "ORI":
            boxes = targets[:, :5].copy()
            labels = targets[:, 5].copy()
        if len(boxes) == 0:

            targets = np.zeros((self.max_labels,  targets.shape[1]), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        
        if self.head_type=="ROT" or self.head_type=="hd":
            boxes_o = targets_o[:, :4]
            boxes_o = xyxy2cxcywh(boxes_o)
            if self.head_type=="ROT":
                ang_o=targets_o[:,4:8]
        elif self.head_type=="ORI":
            boxes_o = targets_o[:, :5]
        if self.head_type=="ROT":
            boxes_o=np.hstack((boxes_o,ang_o))

        labels_o = targets_o[:, targets_o.shape[1]-1]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        


        if random.random() < self.hsv_prob:
            augment_hsv(image)
        if self.head_type=="ROT" or self.head_type=="ORI":
            image_t, boxes = _mirror_R(image, boxes,self.head_type, self.flip_prob)
        else:
            image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_,_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        if self.head_type=="ROT" or self.head_type=="hd":
            boxes = xyxy2cxcywh(boxes)
            mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
            boxes_t = boxes[mask_b]
            labels_t = labels[mask_b]
        else:
            boxes[:,0:4] = r_*boxes[:,0:4]
            boxes_t = boxes
            labels_t = labels
            

        if len(boxes_t) == 0:
            image_t, r_o ,_= preproc(image_o, input_dim)
            boxes_o[:,0:4] = r_o *boxes_o[:,0:4]
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels,targets_o.shape[1] ))
        # a=len(targets_t)
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        # test_img(R_img,np.hstack(( boxes_t,labels_t)))
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ ,_= preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))