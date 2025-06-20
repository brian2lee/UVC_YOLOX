#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import random

import cv2
import numpy as np

from yolox.utils import adjust_box_anns, get_local_rank,obb2hbb_np,aoI_select

from ..data_augment import random_affine
from .datasets_wrapper import Dataset
from yolox.data.datasets import VOC_CLASSES

def test_img(img,labels=None):
    
    cv2.namedWindow("aa",cv2.WINDOW_NORMAL)
    import math
    if labels is not None:
        for box in labels:
            x = np.array(box[0])
            y = np.array(box[1])
            w = np.array(box[2])
            h = np.array(box[3])
            alpha = box[4]
            c = int(box[5])
            x4 = [0.5, -0.5, -0.5, 0.5]
            x4 = x4 * w     
            y4 = [0.5, 0.5, -0.5, -0.5]
            y4 = y4 * h
            alpha = box[4]          
            # alpha= alpha*(2*math.pi)
            corners = np.array([x4,y4])
            sin = math.sin(alpha)
            cos = math.cos(alpha)
            r_matrix=np.array([[cos,-sin],[sin,cos]])
            corners =np.matmul(r_matrix,corners)
            corners[0,:]= x + corners[0,:]
            corners[1,:]= y + corners[1,:]
            corners =corners.T.astype(int)
            color = [255,0,0]
            cv2.line(img,(corners[0][0],corners[0][1]),(corners[1][0],corners[1][1]),color,1)
            cv2.line(img,(corners[1][0],corners[1][1]),(corners[2][0],corners[2][1]),color,1)
            cv2.line(img,(corners[2][0],corners[2][1]),(corners[3][0],corners[3][1]),color,1)
            cv2.line(img,(corners[3][0],corners[3][1]),(corners[0][0],corners[0][1]),color,1)
            v_y=int(0.5*box[3]*math.cos(-box[4]))
            v_x=int(0.5*box[3]*math.sin(-box[4]))

            cv2.line(img,(int(x),int(y)),(int(x-v_x),int(y-v_y)),color,1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text =VOC_CLASSES[int(c)] 
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(
                img,
                (corners[1][0]-10, corners[2][1]-10 ),
                (corners[1][0] + txt_size[0] + -9, corners[2][1] + int(1.5*txt_size[1]-10)),
                [0,0,255],
                -1
            )
            cv2.putText(img, text, (corners[1][0]-10, corners[2][1]-10 + txt_size[1]), font, 0.4, [0,0,0], thickness=1)
            # cv2.putText(img,VOC_CLASSES[int(i[4])],(int(i[0]),int(i[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),1,cv2.LINE_AA)
    cv2.imshow("aa",img)
    cv2.waitKey(0)


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True,
        mosaic_prob=1.0, mixup_prob=1.0, rotate=True,*args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = get_local_rank()
        self.rotate=rotate

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels, _, img_id = self._dataset.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                
                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )
                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    if self.rotate=="ROT" or self.rotate =="hd":
                        labels[:, 0] = scale * _labels[:, 0] + padw
                        labels[:, 1] = scale * _labels[:, 1] + padh
                        labels[:, 2] = scale * _labels[:, 2] + padw
                        labels[:, 3] = scale * _labels[:, 3] + padh
                    else :
                        labels[:, 0] = scale * _labels[:, 0] + padw
                        labels[:, 1] = scale * _labels[:, 1] + padh
                        labels[:, 2] = scale * _labels[:, 2]
                        labels[:, 3] = scale * _labels[:, 3]
                    
                mosaic_labels.append(labels)
            if self.rotate=="ROT":
                if len(mosaic_labels):
                    mosaic_labels = np.concatenate(mosaic_labels, 0)
                    idx=[]
                    for i in range(mosaic_labels.shape[0]):
                        if mosaic_labels[i,0]<0 or mosaic_labels[i,2]> 2 * input_w:
                            idx.append(i)
                        elif mosaic_labels[i,1]<0 or mosaic_labels[i,3]>2 * input_h:
                            idx.append(i)
                        elif mosaic_labels[i,2]-mosaic_labels[i,0]<0.01:
                            idx.append(i)
                        elif mosaic_labels[i,3]-mosaic_labels[i,1]<0.01:
                            idx.append(i)
                        else:
                            pass
                    mosaic_labels=np.delete(mosaic_labels,idx,axis=0)  
            elif self.rotate == "ORI":
                if len(mosaic_labels):

                    mosaic_labels = np.concatenate(mosaic_labels, 0)
                    mosaic_bbox = obb2hbb_np(mosaic_labels)

                    idx=[]
                    for i in range(mosaic_bbox.shape[0]):
                        # if aoI_select(mosaic_bbox[i],[2*input_w,2*input_h],threshold=0.6):
                        #     idx.append(i)
                        if mosaic_bbox[i,0]<0 or mosaic_bbox[i,2]> 2 * input_w:
                            idx.append(i)
                        elif mosaic_bbox[i,1]<0 or mosaic_bbox[i,3]>2 * input_h:
                            idx.append(i)
                        elif mosaic_bbox[i,2]-mosaic_bbox[i,0]<0.01:
                            idx.append(i)
                        elif mosaic_bbox[i,3]-mosaic_bbox[i,1]<0.01:
                            idx.append(i)
                        else:
                            pass
                    # mosaic_labels=mosaic_labels[idx,:] 
                    mosaic_labels=np.delete(mosaic_labels,idx,axis=0)

            else:
                if len(mosaic_labels):
                    mosaic_labels = np.concatenate(mosaic_labels, 0)
                    np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                    np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                    np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                    np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
            # test_img(mosaic_img,mosaic_labels)
            mosaic_img, mosaic_labels = random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
                rotate= self.rotate
                
            )
            
            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if (
                self.enable_mixup
                and not len(mosaic_labels) == 0
                and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
            # test_img(mosaic_img,mosaic_labels)
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            # test_img(mosaic_img,padded_labels)
            img_info = (mix_img.shape[1], mix_img.shape[2])
            # -----------------------------------------------------------------
            # img_info and img_id are not used for training.
            # They are also hard to be specified on a mosaic image.
            # -----------------------------------------------------------------
            return mix_img, padded_labels, img_info, torch.tensor(img_id)

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, img_id = self._dataset.pull_item(idx)
            # test_img(img,label)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels