#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 8
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.mixup_prob = 0.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.0
        self.warmup_epochs = 1
        self.head_type="ORI"
    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import VOCDetection, TrainTransform
        return VOCDetection(
            # data_dir="/data",
            data_dir="/home/rvl224/圖片/box640",
            image_sets=[('train')],
            img_size=self.input_size,
            head_type=self.head_type,
            preproc=TrainTransform(
                max_labels=20,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                head_type=self.head_type),
            cache=cache,
            cache_type=cache_type,
        )

        # return MVTECDetection(
        #     data_dir="/home/rvl224/文件/MVTEC",
        #     image_sets=[('train'),('val')],
        #     img_size=self.input_size,
        #     preproc=TrainTransform(
        #         max_labels=120,
        #         flip_prob=self.flip_prob,
        #         hsv_prob=self.hsv_prob),
        #     cache=cache,
        #     cache_type=cache_type,
        # )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import VOCDetection, ValTransform,MVTECDetection
        legacy = kwargs.get("legacy", False)

        return VOCDetection(
            # data_dir="/data",
            data_dir="/home/rvl224/圖片/box640",
            image_sets=[('test')],
            head_type=self.head_type,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead,OritateHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = OritateHead(self.num_classes, self.width, in_channels=in_channels, act=self.act,depthwise=True)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
