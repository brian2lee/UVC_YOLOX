import os
import os.path
from turtle import pen
import cv2
import numpy as np
from .datasets_wrapper import CacheDataset, cache_read_img
import math
import json
import pickle


class MVTECDetection(CacheDataset):
    def __init__(
        self,
        data_dir,
        image_sets=["train"],
        dataset_name="MVTEC",
        img_size=(416,416),
        path_filename=None,
        cache=False,
        cache_type="ram",
        preproc=None,
        ):
        self.preproc=preproc
        self.root=data_dir
        self._annopath=os.path.join("%s","mvtec_screws_"+"%s.json")        
        self._imgpath=os.path.join("%s","images","%s")
        self.ids=list()
        cats_dict=dict()
        self.all_targets=dict()
        for name in image_sets:
            json_file=open(self._annopath%(data_dir,name),"r")
            data_tree=json.load(json_file)
            images=data_tree["images"]
            for img_info in images:
                del img_info["license"]
            self._find_targets(data_tree["annotations"])
            for image in images:
                self.ids.append(image)

        cats=data_tree['categories']

        self.cats=[{"id":cat["id"]-1,"name":cat["name"]} for cat in cats]
        self.num_imgs=len(self.ids)
        self.img_size = img_size
        # self.cats=cats_dict
        self.annotations = self._load_anno()
        self.name = dataset_name
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=self.root,
            cache_dir_name=f"cache_{self.name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )         
    def __len__(self):
        return self.num_imgs
    def _load_anno(self):
        return [self.load_anno_from_ids(_ids) for _ids in range(self.num_imgs)]
    def load_anno(self, index):
        return self.annotations[index][0]
    def load_anno_from_ids(self,index):
        targets=self.all_targets[self.ids[index]["id"]]
        height,width=self.ids[index]["height"],self.ids[index]["width"]
        bboxes=(box2Tensor(targets)())
        r=min(self.img_size[0]/height,self.img_size[0]/width)
        bboxes[:, :4]*=r
        resized_info = (int(height * r), int(width * r))
        return (bboxes,(height,width),resized_info)
    
    def load_resized_img(self, index):
        img=cv2.imread(self._imgpath%(self.root,self.ids[index]["file_name"]),cv2.IMREAD_COLOR)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img
    def read_img(self, index):
        return self.load_resized_img(index)
    def _find_targets(self,annos):
        for anno in annos:
            _ann=anno["bbox"]
            _ann.append(anno["category_id"]-1)
            if anno["image_id"] in self.all_targets.keys():
                _list=self.all_targets[anno["image_id"]]
                _list.append(_ann)
        # anno_dict.update({anno["image_id"]:anno_dict[anno["image_id"]].append(_ann)})
            else :
                _list=[]
                _list.append(_ann)
                self.all_targets.update({anno["image_id"]:_list})
       


    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        target, img_info, _ = self.annotations[index]
        img = self.read_img(index)
        # for i in target:
        #     cv2.rectangle(img,(int(i[0]-i[2]/2),int(i[1]-i[3]/2)),(int(i[0]+i[2]/2),int(i[1]+i[3]/2)),[255,0,0],3)
        #     # cv2.line(img,(int(xc-bw/2+R1*bw),int(yc-bh/2)),(int(xc-bw/2),int(yc-bh/2+R2*bh)),[0,255,255],2)
        # cv2.imshow("aa",img)
        # cv2.waitKey(0)

        return img, target, img_info, index
    
    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id
    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        IouTh = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        mAPs = []
        for iou in IouTh:
            mAP = self._do_python_eval(output_dir, iou)
            mAPs.append(mAP)

        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        return np.mean(mAPs), mAPs[0]
    def _get_voc_results_file_template(self):
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = os.path.join(self.root, "results")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.cats):
            print("Writing {} ROT results file".format(cls["name"]))
            filename = self._get_voc_results_file_template().format(cls["name"])
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    index = index["file_name"]
                    dets = all_boxes[cls_ind][im_ind]
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                                # dets[k, 4] + 1,
                                # dets[k, 5] + 1,
                                # dets[k, 6] + 1,
                                # dets[k, 7] + 1,
                            )
                        )
    def _do_python_eval(self,output_dir,iou=0.5):
        rootpath= self.root
        cachedir = os.path.join(
            self.root, "annotations_cache"
        )
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        aps = []
        print("Eval IoU : {:.2f}".format(iou))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        ap_dict={}
        for i, cls in enumerate(self.cats):
            npos=0
            filename = self._get_voc_results_file_template().format(cls["name"])
            class_rec={}
            for index in range(self.num_imgs):
                gt_id=self.ids[index]["id"]
                R=[obj for obj in self.all_targets[gt_id] if obj[5]==self.cats[cls["id"]]]
                box_list=box2Tensor(R)()
                if len(R)!=0:
                    det = [False] * len(R)
                    class_rec[self.ids[index]["file_name"]]={"bbox":box_list[:,:8],"det":det}
                    npos=npos+len(R)
                else:
                    class_rec[self.ids[index]["file_name"]]={"bbox":-1*np.ones([1,8]),"det":0}
                    # print("{} is has no {}".format(self.ids[index]["file_name"],cls))
                    continue
            with open(filename, "r") as f:
                lines = f.readlines()
            splitlines = [x.strip().split(" ") for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # sort by confidence
            # print("check")
            sorted_ind = np.argsort(-confidence)
            if BB.size==0:
                ap_dict[cls['name']]={"prec":0,"rec":0,"ap":0}
                print("can't detetect {}".format(cls))
                
            else:
                BB = BB[sorted_ind, :]
                image_ids = [image_ids[x] for x in sorted_ind]

                # go down dets and mark TPs and FPs
                nd = len(image_ids)
                tp = np.zeros(nd)
                fp = np.zeros(nd)

                for d in range(nd):
                    RR=class_rec[image_ids[d]]
                    bb=BB[d, :].astype(float)
                    ovmax = -np.inf
                    BBGT = RR["bbox"].astype(float)
                    if BBGT[0][0] ==-1:
                        ovmax =0
                        continue
                    else:
                        
                        overlaps=self._IoU(bb,BBGT,image_ids[d])
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)
                    if ovmax>iou:
                        if not RR["det"][jmax]:
                            tp[d] = 1.0
                            RR["det"][jmax] = 1
                        else:
                            tp[d]=1.0
                    else:
                        fp[d] = 1.0
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)
                rec=tp / float(npos)
                prec=tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
                mrec = np.concatenate(([0.0], rec, [1.0]))
                mpre = np.concatenate(([0.0], prec, [0.0]))
                for i in range(mpre.size - 1, 0, -1):
                    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
                i = np.where(mrec[1:] != mrec[:-1])[0]
                ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
                ap_dict[cls["id"]]={"prec":prec,"rec":rec,"ap":ap}
        if iou == 0.5:
            for k,v in ap_dict.items():
                print("AP for {} = {:.4f}".format(k,v["ap"]))
                aps +=[v["ap"]]
                if output_dir is not None:
                    with open(os.path.join(output_dir, cls['name']+ "_pr.pkl"), "wb") as f:
                        pickle.dump({"rec": v["rec"], "prec": v["prec"], "ap": v["ap"]}, f)
            print("Mean AP = {:.4f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Results should be very close to the official MATLAB eval code.")
            print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
            print("-- Thanks, The Management")
            print("--------------------------------------------------------------")

        return np.mean(aps)

    def _IoU(self,bb,gt,imgid):
        bb_xmin=bb[0]-0.5*bb[2]
        bb_ymin=bb[1]-0.5*bb[3]
        bb_xmax=bb[0]+0.5*bb[2]
        bb_ymax=bb[1]+0.5*bb[3]
        gt_xmin=gt[:,0]-0.5*gt[:,2]
        gt_ymin=gt[:,1]-0.5*gt[:,3]
        gt_xmax=gt[:,0]+0.5*gt[:,2]
        gt_ymax=gt[:,1]+0.5*gt[:,3]
        ixmin =np.maximum(gt_xmin,bb_xmin)
        iymin =np.maximum(gt_ymin,bb_ymin)
        ixmax =np.minimum(gt_xmax,bb_xmax)
        iymax =np.minimum(gt_ymax,bb_ymax)
        iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
        ih = np.maximum(iymax - iymin + 1.0, 0.0)
        inters = iw * ih       
        # img=cv2.imread(self._imgpath%(self.root,imgid),cv2.IMREAD_COLOR)
        # for i in range(len(gt)):
        #     cv2.rectangle(img,(int(gt_xmin[i]),int(gt_ymin[i])),(int(gt_xmax[i]),int(gt_ymax[i])),[0,255,0],2)
        # cv2.rectangle(img,(int(bb_xmin),int(bb_ymin)),(int(bb_xmax),int(bb_ymax)),[0,0,255],2)
        # cv2.namedWindow("aa",cv2.WINDOW_NORMAL)
        # cv2.imshow("aa",img)
        # cv2.waitKey(1) & 0xff=="q"

        uni=(
            (bb_xmax-bb_xmin+1.0)*(bb_ymax-bb_ymin+1.0)
            +(gt_xmax-gt_xmin+1.0)*(gt_ymax-gt_ymin+1.0)-inters
        )
        overlaps = inters / uni
        return overlaps

               





class box2Tensor():
    def __init__(self,boxes) -> None:
        self.boxes=boxes
        # self.cats =labels_touples

    def __call__(self):
        """
        input:list with "yc xc h w theta label"
        output:[xc,yc,bw,bh,R1,R2,jud,adv,label]
        """
        boxes_list=[]
        #img=cv2.imread("images/screws_002.png")
        for box in self.boxes:
            x=box[1]
            y=box[0]
            w=box[2]
            h=box[3]
            theta=box[4]
            _label=box[5]
            box_points=cv2.boxPoints(((x,y),(w,h),-math.degrees(theta)))
            box_list=self._PointsToDataset(box_points,theta,_label)
            boxes_list.append(box_list)

        boxes_list=np.array(boxes_list,dtype=np.float64)
        return  boxes_list

    def _PointsToDataset(self,Points,theta,_label):
        
        xc=round((Points[0][0]+Points[2][0])/2,3)
        yc=round((Points[0][1]+Points[2][1])/2,3)

        bw=round(np.max(Points,axis=0)[0]-np.min(Points,axis=0)[0],3)
        bh=round(np.max(Points,axis=0)[1]-np.min(Points,axis=0)[1],3)

        R1=(Points[Points.argmin(axis=0)[1]][0]-np.min(Points,axis=0)[0])/bw
        R1=round(R1,5)
        R2=(Points[Points.argmin(axis=0)[0]][1]-np.min(Points,axis=0)[1])/bh
        R2=round(R2,5)
        jud,adv=self._angle_transfrom(theta)

        # cv2.rectangle(img,(int(xc-bw/2),int(yc-bh/2)),(int(xc+bw/2),int(yc+bh/2)),[255,0,0],3)
        # cv2.line(img,(int(xc-bw/2+R1*bw),int(yc-bh/2)),(int(xc-bw/2),int(yc-bh/2+R2*bh)),[0,255,255],2)

        return [xc,yc,bw,bh,_label]


    def _angle_transfrom(self,theta)->int:
        """
        transform Points to jud & adv
        """
        if theta>math.pi:
            theta=theta-2*math.pi
        if theta<=0 and theta>(-0.5*math.pi):
            return [1,1]

        elif theta<=(-0.5*math.pi) and theta>(-math.pi) :
            return [0,1]

        elif (theta>(0.5*math.pi) and theta<=(math.pi)) or theta==-3.141593 :
            return [0,0]
        
        elif theta>0 and theta<=(0.5*math.pi):
            return [1,0]
