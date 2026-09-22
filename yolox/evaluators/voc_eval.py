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

import numpy as np

from yolox.utils import obb2hbb_np, obb2poly_np


def parse_rec(filename):
    """
    Parse a PASCAL VOC XML file.

    Supports two annotation formats:

    1. Rotated object:
       xmin/ymin/xmax/ymax
       R1/R2/jud/adv

    2. Standard rotated box:
       cx/cy/w/h
       angle
    """

    tree = ET.parse(filename)
    objects = []

    for obj in tree.findall("object"):

        obj_struct = {}

        obj_struct["name"] = obj.find("name").text
        obj_struct["difficult"] = int(obj.find("difficult").text)

        bbox = obj.find("bndbox")

        # ---------------------------------------------------------
        # Format 1:
        #
        # xmin ymin xmax ymax
        # R1 R2 jud adv
        # ---------------------------------------------------------
        if bbox.find("xmin") is not None:

            obj_struct["bbox"] = [
                int(bbox.find("xmin").text),
                int(bbox.find("ymin").text),
                int(bbox.find("xmax").text),
                int(bbox.find("ymax").text),
            ]

            obj_struct["angle"] = [
                float(bbox.find("R1").text),
                float(bbox.find("R2").text),
                float(bbox.find("jud").text),
                float(bbox.find("adv").text),
            ]

        # ---------------------------------------------------------
        # Format 2:
        #
        # cx cy w h angle
        # ---------------------------------------------------------
        else:

            obj_struct["bbox"] = [
                int(bbox.find("cx").text),
                int(bbox.find("cy").text),
                int(bbox.find("w").text),
                int(bbox.find("h").text),
            ]

            obj_struct["angle"] = [
                float(bbox.find("angle").text),
            ]

        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall.

    If use_07_metric is True, use VOC 2007 11-point AP.
    Otherwise use the continuous AP calculation.
    """

    if use_07_metric:

        # 11 point metric
        ap = 0.0

        for t in np.arange(0.0, 1.1, 0.1):

            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])

            ap += p / 11.0

    else:

        # Append sentinel values
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # Precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Points where recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # Area under PR curve
        ap = np.sum(
            (mrec[i + 1] - mrec[i]) * mpre[i + 1]
        )

    return ap


def _rotated_iou(gt_box, det_box):
    """
    Calculate rotated IoU using DOTA polygon IoU.

    gt_box:
        [x1, y1, x2, y2, angle...]

    det_box:
        [x1, y1, x2, y2, angle...]
    """

    gt_poly = obb2poly_np(
        np.array([gt_box])
    ).reshape(8)

    det_poly = obb2poly_np(
        np.array([det_box])
    ).reshape(8)

    iou = polyiou.iou_poly(
        polyiou.VectorDouble(gt_poly),
        polyiou.VectorDouble(det_poly),
    )

    return float(iou)


def _calculate_angle_similarity(det_angle, gt_angle):
    """
    Calculate cosine similarity between predicted and GT angles.

    For the UVC representation:

        prediction = [sin(theta), cos(theta)]

    Therefore:

        CS = sin(theta_p) * sin(theta_gt)
           + cos(theta_p) * cos(theta_gt)

    The prediction is normalized so that the metric is robust to
    the magnitude of the predicted vector.

    Returns:
        angle similarity in approximately [-1, 1].
    """

    det_angle = np.asarray(det_angle, dtype=float)
    gt_angle = np.asarray(gt_angle, dtype=float)

    if det_angle.size < 2:
        return 0.0

    t_sin = det_angle[0]
    t_cos = det_angle[1]

    denominator = np.sqrt(
        t_sin ** 2 + t_cos ** 2
    )

    denominator = max(denominator, 1e-8)

    gt_sin = np.sin(gt_angle)
    gt_cos = np.cos(gt_angle)

    similarity = (
        t_sin * gt_sin
        + t_cos * gt_cos
    ) / denominator

    return similarity


def voc_eval(
    detpath,
    annopath,
    imagesetfile,
    classname,
    cachedir,
    ovthresh=0.5,
    use_07_metric=False,
):
    """
    VOC evaluation for rotated object detection.

    Returns:

        rec
            Recall array.

        prec
            Precision array.

        ap
            Detection AP.

        cs
            Average angle cosine similarity over correctly
            detected objects.
    """

    # ============================================================
    # 1. Load ground-truth annotations
    # ============================================================

    if not os.path.isdir(cachedir):
        os.makedirs(cachedir, exist_ok=True)

    cachefile = os.path.join(
        cachedir,
        "annots.pkl"
    )

    # Read image list
    with open(imagesetfile, "r") as f:
        lines = f.readlines()

    imagenames = [
        x.strip()
        for x in lines
    ]

    # Load / create annotation cache
    if not os.path.isfile(cachefile):

        recs = {}

        for i, imagename in enumerate(imagenames):

            recs[imagename] = parse_rec(
                annopath.format(imagename)
            )

            if i % 100 == 0:
                print(
                    "Reading annotation for "
                    "{}/{}".format(
                        i + 1,
                        len(imagenames)
                    )
                )

        print(
            "Saving cached annotations to {}".format(
                cachefile
            )
        )

        with open(cachefile, "wb") as f:
            pickle.dump(recs, f)

    else:

        with open(cachefile, "rb") as f:
            recs = pickle.load(f)

    # ============================================================
    # 2. Extract GT objects for this class
    # ============================================================

    class_recs = {}

    npos = 0

    for imagename in imagenames:

        R = [
            obj
            for obj in recs[imagename]
            if obj["name"] == classname
        ]

        if len(R) > 0:

            bbox = np.array(
                [x["bbox"] for x in R],
                dtype=float,
            )

            ang = np.array(
                [x["angle"] for x in R],
                dtype=float,
            )

            difficult = np.array(
                [x["difficult"] for x in R]
            ).astype(bool)

        else:

            bbox = np.empty(
                (0, 4),
                dtype=float,
            )

            ang = np.empty(
                (0, 1),
                dtype=float,
            )

            difficult = np.empty(
                (0,),
                dtype=bool,
            )

        det = [False] * len(R)

        npos += np.sum(~difficult)

        class_recs[imagename] = {
            "bbox": bbox,
            "difficult": difficult,
            "det": det,
            "angle": ang,
        }

    # ============================================================
    # 3. Read detection results
    # ============================================================

    detfile = detpath.format(classname)

    with open(detfile, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:

        return (
            np.array([]),
            np.array([]),
            0.0,
            0.0,
        )

    # IMPORTANT:
    #
    # split() instead of split(" ")
    #
    # This handles multiple spaces correctly.
    #
    splitlines = [
        x.strip().split()
        for x in lines
        if x.strip()
    ]

    image_ids = [
        x[0]
        for x in splitlines
    ]

    confidence = np.array(
        [
            float(x[1])
            for x in splitlines
        ]
    )

    BB = np.array(
        [
            [
                float(z)
                for z in x[2:]
            ]
            for x in splitlines
        ],
        dtype=float,
    )

    # ============================================================
    # 4. Sort detections by confidence
    # ============================================================

    sorted_ind = np.argsort(
        -confidence
    )

    BB = BB[sorted_ind, :]

    image_ids = [
        image_ids[x]
        for x in sorted_ind
    ]

    # ============================================================
    # 5. Evaluate detections
    # ============================================================

    nd = len(image_ids)

    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # Store angle similarity of matched detections
    angle_similarities = []

    for d in range(nd):

        image_id = image_ids[d]

        R = class_recs[image_id]

        # --------------------------------------------------------
        # Detection box
        # --------------------------------------------------------

        bb = BB[d, :4].astype(float)

        # Everything after the first 4 coordinates is angle data.
        ang = BB[
            d,
            4:BB.shape[1]
        ].astype(float)

        # --------------------------------------------------------
        # Ground-truth boxes
        # --------------------------------------------------------

        BBGT = R["bbox"].astype(float)

        ANGGT = R["angle"].astype(float)

        ovmax = -np.inf
        jmax = -1

        # --------------------------------------------------------
        # Convert detection / GT to HBB
        # --------------------------------------------------------

        if BB.shape[1] == 8:

            # Already HBB
            bb_h = bb

            BBGT_H = BBGT

        elif BB.shape[1] == 5:

            # Rotated box:
            #
            # [cx, cy, w, h, angle]
            #
            # Convert to horizontal bounding box for candidate
            # matching.

            if BBGT.size > 0:

                bb_h = obb2hbb_np(
                    np.concatenate(
                        ([bb], [ang]),
                        axis=1,
                    )
                )[0]

                BBGT_H = obb2hbb_np(
                    np.concatenate(
                        (
                            R["bbox"].astype(float),
                            R["angle"].astype(float),
                        ),
                        axis=1,
                    )
                )

            else:

                bb_h = None
                BBGT_H = None

        else:

            raise ValueError(
                "Unsupported detection format: "
                "BB.shape = {}".format(
                    BB.shape
                )
            )

        # ========================================================
        # 6. Calculate candidate HBB IoU
        # ========================================================

        if BBGT.size > 0:

            ixmin = np.maximum(
                BBGT_H[:, 0],
                bb_h[0],
            )

            iymin = np.maximum(
                BBGT_H[:, 1],
                bb_h[1],
            )

            ixmax = np.minimum(
                BBGT_H[:, 2],
                bb_h[2],
            )

            iymax = np.minimum(
                BBGT_H[:, 3],
                bb_h[3],
            )

            iw = np.maximum(
                ixmax - ixmin + 1.0,
                0.0,
            )

            ih = np.maximum(
                iymax - iymin + 1.0,
                0.0,
            )

            inters = iw * ih

            uni = (
                (
                    bb_h[2]
                    - bb_h[0]
                    + 1.0
                )
                *
                (
                    bb_h[3]
                    - bb_h[1]
                    + 1.0
                )
                +
                (
                    BBGT_H[:, 2]
                    - BBGT_H[:, 0]
                    + 1.0
                )
                *
                (
                    BBGT_H[:, 3]
                    - BBGT_H[:, 1]
                    + 1.0
                )
                - inters
            )

            # Prevent divide-by-zero
            overlaps_hbb = (
                inters
                /
                np.maximum(
                    uni,
                    1e-12,
                )
            )

            # ====================================================
            # 7. Keep GT boxes whose HBB overlaps detection
            # ====================================================

            BBGT_keep_mask = (
                overlaps_hbb > 0
            )

            BBGT_keep = np.concatenate(
                (
                    R["bbox"].astype(float),
                    R["angle"].astype(float),
                ),
                axis=1,
            )[BBGT_keep_mask, :]

            BBGT_keep_index = np.where(
                BBGT_keep_mask
            )[0]

            # ====================================================
            # 8. Calculate rotated IoU
            # ====================================================

            if len(BBGT_keep) > 0:

                rotated_overlaps = []

                for index in range(
                    len(BBGT_keep)
                ):

                    GT = BBGT_keep[index]

                    # Detection rotated box
                    det_obb = np.concatenate(
                        (
                            bb,
                            ang,
                        )
                    )

                    # GT rotated box
                    gt_obb = GT

                    overlap = _rotated_iou(
                        gt_obb,
                        det_obb,
                    )

                    rotated_overlaps.append(
                        overlap
                    )

                rotated_overlaps = np.asarray(
                    rotated_overlaps,
                    dtype=float,
                )

                ovmax = np.max(
                    rotated_overlaps
                )

                local_jmax = np.argmax(
                    rotated_overlaps
                )

                jmax = BBGT_keep_index[
                    local_jmax
                ]

            # ====================================================
            # 9. Calculate angle similarity
            # ====================================================

            # This is calculated only when the detection has
            # the UVC [sin(theta), cos(theta)] representation.

            if (
                BB.shape[1] == 5
                and BBGT.size > 0
                and ANGGT.shape[1] >= 1
                and len(ang) >= 2
            ):

                # Current UVC evaluation
                #
                # prediction:
                #   [sin(theta), cos(theta)]
                #
                # ground truth:
                #   theta
                #
                # CS:
                #   predicted vector dot GT vector
                #   --------------------------------
                #   predicted vector magnitude

                angle_sim = _calculate_angle_similarity(
                    ang,
                    ANGGT,
                )

            elif (
                BB.shape[1] == 8
                and BBGT.size > 0
                and len(ang) >= 2
                and ANGGT.shape[1] >= 2
            ):

                # ------------------------------------------------
                # Existing R1/R2 angle representation
                # ------------------------------------------------

                det_l = np.sqrt(
                    np.square(
                        ang[1]
                        *
                        (
                            bb[3]
                            - bb[1]
                            + 1.0
                        )
                    )
                    +
                    np.square(
                        ang[0]
                        *
                        (
                            bb[2]
                            - bb[0]
                            + 1.0
                        )
                    )
                )

                gt_l = np.sqrt(
                    np.square(
                        ANGGT[:, 1]
                        *
                        (
                            BBGT[:, 3]
                            - BBGT[:, 1]
                            + 1.0
                        )
                    )
                    +
                    np.square(
                        ANGGT[:, 0]
                        *
                        (
                            BBGT[:, 2]
                            - BBGT[:, 0]
                            + 1.0
                        )
                    )
                )

                ab = gt_l * det_l

                ab = np.clip(
                    ab,
                    1e-7,
                    1e7,
                )

                adb = (
                    (
                        ang[1]
                        *
                        (
                            bb[3]
                            - bb[1]
                            + 1.0
                        )
                    )
                    *
                    (
                        ANGGT[:, 1]
                        *
                        (
                            BBGT[:, 3]
                            - BBGT[:, 1]
                            + 1.0
                        )
                    )
                    +
                    (
                        ang[0]
                        *
                        (
                            bb[2]
                            - bb[0]
                            + 1.0
                        )
                    )
                    *
                    (
                        ANGGT[:, 0]
                        *
                        (
                            BBGT[:, 2]
                            - BBGT[:, 0]
                            + 1.0
                        )
                    )
                )

                angle_sim = adb / ab

            else:

                angle_sim = None

        else:

            angle_sim = None

        # ========================================================
        # 10. Determine TP / FP
        # ========================================================

        if ovmax > ovthresh:

            if not R["difficult"][jmax]:

                if not R["det"][jmax]:

                    tp[d] = 1.0

                    R["det"][jmax] = True

                    # Only matched TP contributes to CS
                    if angle_sim is not None:

                        if np.ndim(angle_sim) == 0:

                            angle_similarities.append(
                                float(angle_sim)
                            )

                        else:

                            angle_similarities.append(
                                float(
                                    angle_sim[jmax]
                                )
                            )

                else:

                    # Multiple detection of the same GT
                    fp[d] = 1.0

        else:

            fp[d] = 1.0

    # ============================================================
    # 11. Precision / Recall
    # ============================================================

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    if npos > 0:

        rec = tp / float(npos)

    else:

        rec = np.zeros_like(tp)

    prec = tp / np.maximum(
        tp + fp,
        np.finfo(np.float64).eps,
    )

    # ============================================================
    # 12. Calculate AP
    # ============================================================

    ap = voc_ap(
        rec,
        prec,
        use_07_metric,
    )

    # ============================================================
    # 13. Calculate CS
    # ============================================================

    if len(angle_similarities) > 0:

        cs = (
            np.sum(angle_similarities)
            /
            len(angle_similarities)
        )

        # Numerical safety
        cs = float(
            np.clip(
                cs,
                -1.0,
                1.0,
            )
        )

    else:

        cs = 0.0

    return rec, prec, ap, cs
