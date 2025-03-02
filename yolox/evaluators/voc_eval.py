#!/usr/bin/env python3
# Code are based on
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# Copyright (c) Bharath Hariharan.
# Copyright (c) Megvii, Inc. and its affiliates.

from operator import le
import os
import pickle
import xml.etree.ElementTree as ET
from DOTA_devkit import polyiou
from yolox.utils import cal_iou
import torch
import numpy as np
from yolox.utils import obb2hbb_np , obb2poly_np

def parse_rec(filename):
    """Parse a PASCAL VOC xml file"""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        # obj_struct["pose"] = obj.find("pose").text
        # obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        if bbox.find("xmin") is not None:
            obj_struct["bbox"] = [
                int(bbox.find("xmin").text),
                int(bbox.find("ymin").text),
                int(bbox.find("xmax").text),
                int(bbox.find("ymax").text),
            ]        
            obj_struct["angle"]= [
                float(bbox.find("R1").text),
                float(bbox.find("R2").text),
                float(bbox.find("jud").text),
                float(bbox.find("adv").text),
            ]
            objects.append(obj_struct)
        else:
            obj_struct["bbox"] = [
                int(bbox.find("cx").text),
                int(bbox.find("cy").text),
                int(bbox.find("w").text),
                int(bbox.find("h").text),
            ]        
            obj_struct["angle"]= [
                float(bbox.find("angle").text),
            ]
            objects.append(obj_struct)
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(
    detpath,
    annopath,
    imagesetfile,
    classname,
    cachedir,
    ovthresh=0.5,
    use_07_metric=False,
):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, "annots.pkl")
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print(f"Reading annotation for {i + 1}/{len(imagenames)}")
        # save
        print(f"Saving cached annotations to {cachefile}")
        with open(cachefile, "wb") as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, "rb") as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    ang_ap = 0.0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        ang=np.array([x["angle"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
  
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det,"angle":ang}


    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return 0, 0, 0

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    ang_ps=[]
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :4].astype(float)
        # bb =[0.5*(BB[d,0]+BB[d,2]),0.5*(BB[d,0]+BB[d,2])]
        ang=BB[d,4:BB.shape[1]].astype(float)
        
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)
        ANGGT=R["angle"].astype(float)
        if BB.shape[1]==8:
            bb_h = bb
            BBGT_H = BBGT
        elif BB.shape[1]==5:
            if BBGT.size >0 :
                bb_h=obb2hbb_np(np.concatenate(([bb],[ang]),axis=1))[0]
                BBGT_H=obb2hbb_np(np.concatenate((R["bbox"].astype(float),R["angle"].astype(float)),axis=1))
        
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT_H[:, 0], bb_h[0])
            iymin = np.maximum(BBGT_H[:, 1], bb_h[1])
            ixmax = np.minimum(BBGT_H[:, 2], bb_h[2])
            iymax = np.minimum(BBGT_H[:, 3], bb_h[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb_h[2] - bb_h[0] + 1.0) * (bb_h[3] - bb_h[1] + 1.0)
                + (BBGT_H[:, 2] - BBGT_H[:, 0] + 1.0) * (BBGT_H[:, 3] - BBGT_H[:, 1] + 1.0) - inters
            )
            if BB.shape[1]==8:
                det_l=np.sqrt(np.square(ang[1]*(bb[3] - bb[1] + 1.0))+np.square(ang[0]*(bb[2] - bb[0] + 1.0)))
                gt_l=np.sqrt(np.square(ANGGT[:,1]*(BBGT[:,3] - BBGT[:,1] + 1.0))+np.square(ANGGT[:,0]*(BBGT[:,2] - BBGT[:,0] + 1.0)))
                ab=gt_l*det_l
                ab=np.clip(ab,1e-7,1e7)
                adb=(ang[1]*(bb[3] - bb[1] + 1.0))*(ANGGT[:,1]*(BBGT[:,3] - BBGT[:,1] + 1.0))+(ang[0]*(bb[2] - bb[0] + 1.0))*(ANGGT[:,0]*(BBGT[:,2] - BBGT[:,0] + 1.0))
            
                ang_p=adb/ab
            if BB.shape[1]==5:
                ang_p = np.cos(np.abs(ANGGT-ang))

                
            overlaps = inters / uni
            BBGT_keep_mask =overlaps > 0
            BBGT_keep =np.concatenate((R["bbox"].astype(float),R["angle"].astype(float)),axis=1)[BBGT_keep_mask,:]
            BBGT_keep_index =np.where(overlaps > 0)[0]
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    obb = np.append(bb,ang)
                    obb_gt = np.array([BBGT_keep[index]])
                    overlap = polyiou.iou_poly(polyiou.VectorDouble(obb2poly_np(obb_gt).reshape(8)), polyiou.VectorDouble(obb2poly_np(np.array([obb])).reshape(8)))
                    
                    overlaps.append(overlap)
                return overlaps

            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                jmax = BBGT_keep_index[jmax]            
            # ovmax = np.max(overlaps)
            # jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                    ang_ps.append(ang_p[jmax])
                else:
                    # ang_ps.append(ang_p[jmax])
                    fp[d] = 1.0
                
        else:
            fp[d] = 1.0

        # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    ang_ap=(np.sum(ang_ps)+1e-8)/(tp[-1]+1e-6)
    ang_ap=np.clip(ang_ap,1e-16,1.0)
    # ang_ap =np.clip(ang_ap,1e-15,1e+10)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap ,ang_ap
