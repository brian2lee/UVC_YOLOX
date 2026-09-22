#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np
import math

__all__ = ["vis"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img



def visR(img, boxes,angles, scores, cls_ids, conf=0.5, class_names=None):
    def decode_box(boxes,angle):
        x = (boxes[2]+boxes[0])/2
        y = (boxes[3]+boxes[1])/2
        W = boxes[2]-boxes[0]
        H = boxes[3]-boxes[1]
        R1 = np.clip(angle[0],0.0001,0.9999)
        R2 = np.clip(angle[1],0.0001,0.9999)
        w1 = np.sqrt(np.power(W*R1,2)+np.power(H*R2,2))
        w2 = np.sqrt(np.power(W*(1-R1),2)+np.power(H*(1-R2),2))
        maxW = np.maximum(w1,w2)
        w1_is_maximun = w1==maxW
        if w1_is_maximun :
            t = 0.5*math.pi+math.atan(-(H*R2)/(W*R1))
        else :
            t = math.atan((H*(1-R2))/(W*(1-R1)))
        
        out_boxes = [x,y,w2,w1,t]
        return out_boxes
    def box2corners_th(box):
        x = box[0].numpy()
        y = box[1].numpy()
        w = box[2].numpy()
        h = box[3].numpy()
        alpha = box[4]
        x4 = [0.5, -0.5, -0.5, 0.5]
        x4 = x4 * w     
        y4 = [0.5, 0.5, -0.5, -0.5]
        y4 = y4 * h

        corners = np.array([x4,y4])
        sin = math.sin(alpha)
        cos = math.cos(alpha)
        r_matrix=np.array([[cos,-sin],[sin,cos]])
        corners =np.matmul(r_matrix,corners)
        corners[0,:]= x + corners[0,:]
        corners[1,:]= y + corners[1,:]
        corners =corners.T.astype(int)
        return corners

    for i in range(len(boxes)):
        box = boxes[i]
        
        cls_id = int(cls_ids[i])
        score = scores[i]
        angle= np.clip(angles[i],0.0001,0.9999)
        rbox=decode_box(box,angle)
        corners= box2corners_th(rbox)
        if angle[2]>0.8:
            jud =1
        else :
            jud =0
        if angle[3]>0.8:
            adv =1
        else :
            adv =0
        # angle point
        cx =int(rbox[0])
        cy =int(rbox[1])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.line(img,(corners[0][0],corners[0][1]),(corners[1][0],corners[1][1]),color,2)
        cv2.line(img,(corners[1][0],corners[1][1]),(corners[2][0],corners[2][1]),color,2)
        cv2.line(img,(corners[2][0],corners[2][1]),(corners[3][0],corners[3][1]),color,2)
        cv2.line(img,(corners[3][0],corners[3][1]),(corners[0][0],corners[0][1]),color,2)
        if jud==1 and adv==1:
            cv2.line(img,(cx,cy),(int(0.5*(corners[2][0]+corners[3][0])),int(0.5*(corners[2][1]+corners[3][1]))),[200,0,200],2)
        elif jud==0 and adv==0:
            cv2.line(img,(cx,cy),(int(0.5*(corners[1][0]+corners[0][0])),int(0.5*(corners[1][1]+corners[0][1]))),[200,0,200],2)
        elif jud==1 and adv==0:
            cv2.line(img,(cx,cy),(int(0.5*(corners[2][0]+corners[1][0])),int(0.5*(corners[2][1]+corners[1][1]))),[200,0,200],2)
        elif jud==0 and adv==1:
            cv2.line(img,(cx,cy),(int(0.5*(corners[0][0]+corners[3][0])),int(0.5*(corners[0][1]+corners[3][1]))),[200,0,200],2)      

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (corners[1][0]-10, corners[2][1]-10 ),
            (corners[1][0] + txt_size[0] + -9, corners[2][1] + int(1.5*txt_size[1]-10)),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (corners[1][0]-10, corners[2][1]-10 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def visO(img, boxes, scores, cls_ids, conf=0.5, class_names=None,gt=True):
    def box2corners_th(box):
        if not box.dtype== np.float32:
            x = box[0].numpy()
            y = box[1].numpy()
            w = box[2].numpy()
            h = box[3].numpy()
            alpha = box[4]
        else:
            x = np.array(box[0])
            y = np.array(box[1])
            w = np.array(box[2])
            h = np.array(box[3])
            alpha = box[4]          
            # alpha=(alpha-0.1)/0.8
            # alpha= alpha*(2*math.pi)
        x4 = [0.5, -0.5, -0.5, 0.5]
        x4 = x4 * w     
        y4 = [0.5, 0.5, -0.5, -0.5]
        y4 = y4 * h

        corners = np.array([x4,y4])
        sin = math.sin(alpha)
        cos = math.cos(alpha)
        r_matrix=np.array([[cos,-sin],[sin,cos]])
        corners =np.matmul(r_matrix,corners)
        corners[0,:]= x + corners[0,:]
        corners[1,:]= y + corners[1,:]
        corners =corners.T.astype(int)
        return corners

    for i in range(len(boxes)):
        box = boxes[i]
        if not gt:
            cls_id = int(cls_ids[i])
            score = scores[i]
        corners= box2corners_th(box)
        # angle point
        cx =int(box[0])
        cy =int(box[1])
        if gt :
            color = [0,255,0]
        else:
            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.line(img,(corners[0][0],corners[0][1]),(corners[1][0],corners[1][1]),color,2)
        cv2.line(img,(corners[1][0],corners[1][1]),(corners[2][0],corners[2][1]),color,2)
        cv2.line(img,(corners[2][0],corners[2][1]),(corners[3][0],corners[3][1]),color,2)
        cv2.line(img,(corners[3][0],corners[3][1]),(corners[0][0],corners[0][1]),color,2)
        v_y=int(0.5*box[3]*math.cos(-box[4]))
        v_x=int(0.5*box[3]*math.sin(-box[4]))
        cv2.line(img,(cx,cy),(cx-v_x,cy-v_y),color,2)
        if not gt:
            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (corners[1][0]-10, corners[2][1]-10 ),
                (corners[1][0] + txt_size[0] + -9, corners[2][1] + int(1.5*txt_size[1]-10)),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (corners[1][0]-10, corners[2][1]-10 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        # 0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
