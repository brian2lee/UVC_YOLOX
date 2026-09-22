

import os
import random
import shutil
import cv2 
import math
import numpy as np
import json
import datetime
import xml.etree.ElementTree as ET
import torch



def bbox2VOC(bbox):
    x=bbox[0]
    y=bbox[1]
    w=bbox[2]
    h=bbox[3]
    theta=bbox[4]
    if theta > 6.283 or theta<0.00001:
        theta=0.00001
    # theta=(bbox[4]+math.pi)%(2*math.pi)
    Points=cv2.boxPoints(((x,y),(w,h),math.degrees(theta)))
    x1=round(np.min(Points,axis=0)[0],0)
    x2=round(np.max(Points,axis=0)[0],0)
    y1=round(np.min(Points,axis=0)[1],0)
    y2=round(np.max(Points,axis=0)[1],0)
    if (theta%(0.5*math.pi)<0.01):
        R1 = 0.0001
        R2 = 0.9999
    else:

        R1=(Points[Points.argmin(axis=0)[1]][0]-np.min(Points,axis=0)[0])/(x2-x1)
        R1=np.clip(R1,0.0001,0.9999)
        R1=round(R1,5)
        R2=(Points[Points.argmin(axis=0)[0]][1]-np.min(Points,axis=0)[1])/(y2-y1)
        R2=np.clip(R2,0.0001,0.9999)
        R2=round(R2,5)
    return [x1,x2,y1,y2,R1,R2]

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
a= [10,20,20,10,0.0]
box = bbox2VOC(a)

def _draw(img,bbox):
    img=cv2.imread(img)
    px1=int(bbox[0]+bbox[4]*(bbox[1]-bbox[0]))
    py1=int(bbox[2])
    px2=int(bbox[0])
    py2=int(bbox[2]+bbox[5]*(bbox[3]-bbox[2]))


    cv2.rectangle(img,(bbox[0],bbox[2]),(bbox[1],bbox[3]),[255,255,0],3)
    # cv2.line(img,(230,3196),(464,1615),[255,0,0],3)
    cv2.line(img,(px2,py2),(px1,py1),[255,0,0],3)
    cv2.namedWindow("aaa",cv2.WINDOW_NORMAL)
    cv2.imshow("aaa",img)
    cv2.waitKey(0)




def box2trainingtype(input_dir_set,t_type,count):
    """
    input type is xml: loading /path/to/dir whitch save file.xml
    """
    img_list=[]
    tonow =datetime.datetime.now()
    target_dir =os.path.join(os.getcwd(),"output",tonow.strftime('%m%d_%H%M'))
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    os.mkdir(os.path.join(target_dir,"Annotations"))
    os.mkdir(os.path.join(target_dir,"JPEGImages"))
    for input_dir in input_dir_set:
        for file in os.listdir(input_dir):
            if file.endswith(".xml"):
                file_path=os.path.join(input_dir,file)
                tree = ET.parse(file_path)
                root_i=tree.getroot()
                path=root_i.find("path").text
                path=path.split("/")[-1]
                s=root_i.find("size")
                img_w=s.find("width").text
                img_h=s.find("height").text
                


                #write xml
                root_o=ET.Element("annotation")
                filename= ET.SubElement(root_o,"filename")
                filename.text=str(count)+".png"
                size=ET.SubElement(root_o,"size")
                width=ET.SubElement(size,"width")
                width.text=str(img_w)
                height=ET.SubElement(size,"height")
                height.text=str(img_h)
                depth=ET.SubElement(size,"depth")
                depth.text=str(3)


                for obj in root_i.findall("object"):
                    box= obj.find("robndbox")
                    cx=float(box.find("cx").text)
                    cy=float(box.find("cy").text)
                    w= float(box.find("w").text)
                    h= float(box.find("h").text)
                    angle= float(box.find("angle").text)
                    label=obj.find("name").text
                    
                    if label =="M6":
                        ll=[cx,cy,w,h,angle,"M8m"]
                    else :
                        ll =[cx,cy,w,h,angle,label]
                    if t_type == "R":
                        xxyyR=bbox2VOC(ll[:5])
                        # _draw('/home/rvl224/文件/paperbox/img/IMG_20230512_144131_BURST6.jpg',xxyyR)
                        #write bndbox
                        object=ET.SubElement(root_o,"object")
                        name=ET.SubElement(object,"name")
                        name.text=ll[5]
                        defficult=ET.SubElement(object,"difficult")
                        defficult.text=str(0)
                        bndbox=ET.SubElement(object,"bndbox")
                        xmin=ET.SubElement(bndbox,"xmin")
                        xmin.text=str(int(xxyyR[0]))
                        ymin=ET.SubElement(bndbox,"ymin")
                        ymin.text=str(int(xxyyR[2]))
                        xmax=ET.SubElement(bndbox,"xmax")
                        xmax.text=str(int(xxyyR[1]))
                        ymax=ET.SubElement(bndbox,"ymax")
                        ymax.text=str(int(xxyyR[3]))
                        R1=ET.SubElement(bndbox,"R1")
                        R1.text=str(xxyyR[4])
                        R2=ET.SubElement(bndbox,"R2")
                        R2.text=str(xxyyR[5])
                        the=float(ll[4])
                        jud,adv=_angle_transform(the)
                        jud_=ET.SubElement(bndbox,"jud")
                        jud_.text=str(jud)
                        adv_=ET.SubElement(bndbox,"adv")
                        adv_.text=str(adv)
                    elif t_type =="theta":
                        object=ET.SubElement(root_o,"object")
                        name=ET.SubElement(object,"name")
                        name.text=ll[5]  
                        defficult=ET.SubElement(object,"difficult")
                        defficult.text=str(0)
                        bndbox=ET.SubElement(object,"bndbox")
                        cx=ET.SubElement(bndbox,"cx")
                        cx.text=str(int(ll[0]))
                        cy=ET.SubElement(bndbox,"cy")
                        cy.text=str(int(ll[1]))
                        w=ET.SubElement(bndbox,"w")
                        w.text=str(int(ll[2]))
                        h=ET.SubElement(bndbox,"h")
                        h.text=str(int(ll[3]))
                        angle=ET.SubElement(bndbox,"angle")
                        angle.text=str(ll[4])                                    
                tree=ET.ElementTree(root_o)
                


                
                dir=os.path.join(target_dir,"Annotations",str(count)+".xml")
                img =cv2.imread(os.path.join(input_dir,path[0:-4]+".jpg"))
                cv2.imwrite(os.path.join(target_dir,"JPEGImages",str(count)+".png"),img)
                tree.write(dir)
                img_list.append(count)
                print("Xml files save in {}".format(dir))
                count =count +1
        # print("{} convert to dataset,total number of image is {}".format(input_dir,count))

    f=open(os.path.join(target_dir,"data.txt"),"w+")
    for i in img_list:
        f.write("{}\n".format(i))
    f.close()
    f=open(os.path.join(target_dir,"include_source_dir.txt"),"w+")
    for i in input_dir_set:
        f.write("{}\n".format(i))
    f.write("total images is{}\n".format(count-1))
    f.close()
    # print("image set save in {}".format(os.path.join(target_dir,"data.txt")))


    # print(tt)
# txt=open("/home/rvl224/文件/MVTEC/txt/screws_002.txt")
# img=cv2.imread("/home/rvl224/文件/MVTEC/images/screws_002.png")

def dataset_aug(xmldir,num):
    """
    add data_to_dataset
    """
    outdir=os.path.join(xmldir,"output")
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for file in os.listdir(xmldir):
        if file.endswith(".xml"):
            file_path=os.path.join(xmldir,file)
            tree = ET.parse(file_path)
            root_i=tree.getroot()
            filename=root_i.find("filename")
            filename.text=str(num)+".jpg"
            path=root_i.find("path")
            path.text=os.path.join(outdir,str(num)+".jpg")
            tree.write(os.path.join(outdir,str(num)+".xml"))
            
            img=cv2.imread(os.path.join(xmldir,file[:-3]+"jpg"))
            cv2.imwrite(os.path.join(outdir,str(num)+".png"),img)           
            num=num+1
def class_num(img_set,ann_file):
    """
    input: img_set: *.txt
           ann_file : *.xml
    
    return:
        print(class num)
    """
    f =open(img_set)
    class_dict=dict()
    for line in f.readlines():
        xmlfile = os.path.join(ann_file,line[:-1]+".xml")
        tree = ET.parse(xmlfile)
        root=tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name  not in class_dict.keys():
                class_dict[name] =0
            class_dict[name] =class_dict[name] +1
    print(class_dict)

# a=vis("/home/rvl224/文件/wilbur_data/VOC",None)
# a._show()
# dataset_aug("/home/rvl224/文件/wilbur_data/aaa/0531",121)

# cv2.namedWindow("aa",cv2.WINDOW_NORMAL)
def test_dataset():
    img=cv2.imread("/home/rvl224/文件/MVTEC/VOC/JPEGImages/screws_315.png")
    for i in range(401,800):
        rand_t = random.randint(0,361)
        rand_s = random.uniform(1.0,1.4)
        M =cv2.getRotationMatrix2D((1611,813),rand_t,rand_s)
        res =cv2.warpAffine(img,M,(1920,1440),borderValue=[113,186,247])
        res =res[573:1053,1371:1851,:]
        x=np.array([240])
        y=np.array([240])
        w_ =np.array([47*rand_s])
        h_= np.array([139*rand_s])
        t =(0.262-(rand_t/180)*math.pi)%(2*math.pi)
        cv2.imwrite(str(i)+".png",res)
        root_o=ET.Element("annotation")
        filename= ET.SubElement(root_o,"filename")
        filename.text=str(i)+".png"
        size=ET.SubElement(root_o,"size")
        width=ET.SubElement(size,"width")
        width.text=str(480)
        height=ET.SubElement(size,"height")
        height.text=str(480)
        depth=ET.SubElement(size,"depth")
        depth.text=str(3)
        object=ET.SubElement(root_o,"object")
        name=ET.SubElement(object,"name")
        name.text="type_012"  
        defficult=ET.SubElement(object,"difficult")
        defficult.text=str(0)
        bndbox=ET.SubElement(object,"bndbox")
        cx=ET.SubElement(bndbox,"cx")
        cx.text=str(int(x[0]))
        cy=ET.SubElement(bndbox,"cy")
        cy.text=str(int(y[0]))
        w=ET.SubElement(bndbox,"w")
        w.text=str(int(w_[0]))
        h=ET.SubElement(bndbox,"h")
        h.text=str(int(h_[0]))
        angle=ET.SubElement(bndbox,"angle")
        angle.text=str(t)                                    
        tree=ET.ElementTree(root_o)
        tree.write(str(i)+".xml")
# test_dataset()
# f= open("test.txt","a")
# for i in range(751,800):
#     f.write(str(i)+" \n")
# f.close()

# a= float(0)
# file_list =["/home/rvl224/文件/wilbur_data/source/0807_3","/home/rvl224/文件/wilbur_data/source/0531",
#             "/home/rvl224/文件/wilbur_data/source/0521","/home/rvl224/文件/wilbur_data/source/07071",
#             "/home/rvl224/文件/wilbur_data/source/0807_2","/home/rvl224/文件/wilbur_data/source/0807_1"]
file_list =["/home/rvl224/圖片/box"]
type_ = "theta"
box2trainingtype(file_list,type_,1)

# class_num("/home/rvl224/文件/wilbur_data/VOC/ImageSets/test.txt","/home/rvl224/文件/wilbur_data/VOC/Annotations")
# a= (10*math.pi/180)*math.cos(10*math.pi)

# print(a)