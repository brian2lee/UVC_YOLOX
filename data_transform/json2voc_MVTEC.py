import os
import cv2 
import math
import numpy as np
import json

import xml.etree.ElementTree as ET



def bbox2VOC(bbox):
    x=bbox[0]
    y=bbox[1]
    w=bbox[2]
    h=bbox[3]
    theta=bbox[4]
    # if theta > 6.283 or theta<0.00001:
    #     theta=0.00001
    # # theta=(bbox[4]+math.pi)%(2*math.pi)
    # Points=cv2.boxPoints(((x,y),(w,h),math.degrees(theta)))
    # x1=round(np.min(Points,axis=0)[0],0)
    # x2=round(np.max(Points,axis=0)[0],0)
    # y1=round(np.min(Points,axis=0)[1],0)
    # y2=round(np.max(Points,axis=0)[1],0)
    # if (theta%(0.5*math.pi)<0.01):
    #     R1 = 0.001
    #     R2 = 0.999
    # else:

    #     R1=(Points[Points.argmin(axis=0)[1]][0]-np.min(Points,axis=0)[0])/(x2-x1)
    #     R1=np.clip(R1,0.001,0.999)
    #     R1=round(R1,5)
    #     R2=(Points[Points.argmin(axis=0)[0]][1]-np.min(Points,axis=0)[1])/(y2-y1)
    #     R2=np.clip(R2,0.001,0.999)
    #     R2=round(R2,5)
    return [x,y,w,h,theta]

def _angle_transform(theta)->int:
        """
        transform Points to jud & adv
        """
        theta=theta%(2*math.pi)

        if theta >= 0.0 and theta < (0.5*math.pi):
            return [1,1]

        elif theta >= (0.5*math.pi) and theta < (math.pi) :
            return [0,1]

        elif theta >= (math.pi) and theta < (1.5*math.pi):
            return [0,0]
        
        elif theta >= (1.5*math.pi) :
            return [1,0]


class box2VOC():
    def __init__(self,boxes) -> None:
        self.boxes=boxes
        # self.cats =labels_touples

    def __call__(self,type_):
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
            if type_ =="txt":
                box_points=cv2.boxPoints(((x,y),(w,h),-math.degrees(theta)))
                boxes_list.append((np.reshape(box_points,8),_label)) 
            # box_list=self._PointsToDataset(box_points,theta,_label)
            
            elif type_ =="xml":
                boxes_list.append([x,y,w,h,theta,_label])
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

def to_yolo_txt(type_i,
                type_o, 
                dir_name,
                set_path):
    """
    input type is json: loading /path/to/file.json
    input type is xml: loading /path/to/dir whitch save file.xml

    """
    
    assert set_path[-4:]==type_i ,"Type of file shold be *.{} ,file name is {}".format(type_i,set_path)


    ann_path=set_path
    json_file=open(ann_path,"r")
    text_file_=dir_name
    data_tree=json.load(json_file)
    if type_o=="txt":
        VOC_CLASSES = (
            "type_001",
            "type_002",
            "type_003",
            "type_004",
            "type_005",
            "type_006",
            "type_007",
            "type_008",
            "type_009",
            "type_010",
            "type_011",
            "type_012",
            "type_013",
        )


        images=data_tree["images"]
        for img_info in images:
            bb=[]
            txt_path=open(os.path.join(text_file_,img_info["file_name"][0:-4]+".txt"),"w+")
            for i in data_tree["annotations"]:
                if i["image_id"]==img_info["id"]:
                    i["bbox"].append(VOC_CLASSES[i["category_id"]-1])
                    bb.append(i["bbox"])   
            tt=box2VOC(bb)(type_o)
            for i in tt:
                box =i[0]
                _label =i[1]
                txt_path.write("{:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {} {}\n"
                                .format(box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7],_label,int(0)))
            txt_path.close()
    elif type_o=="xml":
        images=data_tree["images"]
        cats=data_tree["categories"]
        cat_l=[]
        cat_l=[{"id":i["id"],"name":i["name"]} for i in cats]
        img_list=[]
        for img_inf in images:
            
            root=ET.Element("annotation")
            filename= ET.SubElement(root,"filename")
            filename.text=img_inf["file_name"]
            size=ET.SubElement(root,"size")
            width=ET.SubElement(size,"width")
            width.text=str(img_inf["width"])
            height=ET.SubElement(size,"height")
            height.text=str(img_inf["height"])
            depth=ET.SubElement(size,"depth")
            depth.text=str(3)
            for i in data_tree["annotations"]:
                if i["image_id"]==img_inf["id"]:
                    object=ET.SubElement(root,"object")
                    name=ET.SubElement(object,"name")
                    name.text=cat_l[i["category_id"]-1]["name"]
                    difficult=ET.SubElement(object,"difficult")
                    difficult.text=str(0)
                    bndbox=ET.SubElement(object,"bndbox")
                    bbox=i["bbox"]
                    xxyyR=bbox2VOC(bbox)

                    cx=ET.SubElement(bndbox,"cx")
                    cx.text=str(int(xxyyR[1]))
                    cy=ET.SubElement(bndbox,"cy")
                    cy.text=str(int(xxyyR[0]))
                    w=ET.SubElement(bndbox,"w")
                    w.text=str(int(xxyyR[3]))
                    h=ET.SubElement(bndbox,"h")
                    h.text=str(int(xxyyR[2]))
                    a = ET.SubElement(bndbox,"angle")
                    a.text=str(round(float(0.8*(((-xxyyR[4]+0.5*math.pi)%(2*math.pi))/(2*math.pi)))+0.1,4))
            tree=ET.ElementTree(root)
            dir=os.path.join(text_file_,img_inf["file_name"][0:-4]+".xml")
            tree.write(dir)
            img_list.append(img_inf["file_name"][0:-4])
        f=open(set_path[0:-4]+"txt","w+")
        for i in img_list:
            f.write("{}\n".format(i))
        f.close()

def num_class(set_path):
    """
    input type is json: loading /path/to/file.json
    input type is xml: loading /path/to/dir whitch save file.xml

    """
    train_path = set_path +"/mvtec_screws_train.json"
    test_path = set_path +"/mvtec_screws_test.json"
    val_path = set_path +"/mvtec_screws_val.json"


    train_file=open(train_path,"r")
    test_file=open(test_path,"r")
    val_file=open(val_path,"r")
    train_tree=json.load(train_file)
    val_tree = json.load(val_file)
    test_tree=json.load(test_file)
    VOC_CLASSES = [
        "type_001",
        "type_002",
        "type_003",
        "type_004",
        "type_005",
        "type_006",
        "type_007",
        "type_008",
        "type_009",
        "type_010",
        "type_011",
        "type_012",
        "type_013",
    ]
    Num_test = np.zeros(13,dtype=np.uint32)
    Num_train = np.zeros(13,dtype=np.uint32)
    for i in test_tree["annotations"]:
        Num_test[i["category_id"]-1]=Num_test[i["category_id"]-1]+1
    for i in train_tree["annotations"]:
        Num_train[i["category_id"]-1]=Num_train[i["category_id"]-1]+1
    for i in val_tree["annotations"]:
        Num_train[i["category_id"]-1]=Num_train[i["category_id"]-1]+1
    return VOC_CLASSES,Num_train,Num_test
# to_yolo_txt(type_i="json",type_o="xml",dir_name="/home/rvl224/文件/MVTEC",set_path="/home/rvl224/文件/MVTEC/mvtec_screws_test.json")
clss,train_data,test_data=num_class(set_path="/home/rvl224/文件/MVTEC")

print(clss)
print(train_data.tolist())
print(train_data.sum())
print(test_data.tolist())
