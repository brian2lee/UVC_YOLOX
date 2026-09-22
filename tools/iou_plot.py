import math
from yolox.utils import cal_iou,bboxes_iou,box2corners_th,poly2hdd
import torch
from matplotlib import pyplot as plt

boxA = []
boxB = torch.Tensor([[1000,1000,50,250,0]])

for i in range(91):
    boxA.append([1000,1000,50,250,(i/180)*math.pi])
boxA = torch.Tensor(boxA)
pair_Piou=cal_iou(boxA,boxB)

boxA_h = box2corners_th(boxA)
boxA_h = poly2hdd(boxA_h)

boxB_h = box2corners_th(boxB)
boxB_h = poly2hdd(boxB_h)

pair_iou = bboxes_iou(boxA_h,boxB_h)
pair_Piou = pair_Piou.cpu().numpy()
pair_iou = pair_iou.numpy()

print(pair_iou-pair_Piou)