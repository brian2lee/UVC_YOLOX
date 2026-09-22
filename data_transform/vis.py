import math
import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
class vis():
    def __init__(self,RootPath,ImgPath=None):
        if ImgPath!= None :
            self.root=RootPath
            self.img_path="{}/JPEGImages/{}".format(RootPath,ImgPath)
            self.xml_path="{}/Annotations/{}".format(RootPath,ImgPath[:-3]+"xml")

        else :
            self.root=RootPath
            self.img_path=os.path.join(RootPath,"JPEGImages")
            self.xml_path=os.path.join(RootPath,"Annotations")

    def load_xml(self,file):
        tree = ET.parse(file)
        root=tree.getroot()
        objs=[]

        for obj in root.findall("object"):
            cls=str(obj.find("name").text)
            bndbox= obj.find("bndbox")
            xmin=float(bndbox.find("xmin").text)
            ymin=float(bndbox.find("ymin").text)
            xmax=float(bndbox.find("xmax").text)
            ymax=float(bndbox.find("ymax").text)
            R1=float(bndbox.find("R1").text)
            R2=float(bndbox.find("R2").text)
            jud=int(bndbox.find("jud").text)
            adv=int(bndbox.find("adv").text)
            objs.append([cls,xmin,ymin,xmax,ymax,R1,R2,jud,adv])
        return objs
    def load_xml_o(self,file):
        tree = ET.parse(file)
        root=tree.getroot()
        objs=[]

        for obj in root.findall("object"):
            cls=str(obj.find("name").text)
            bndbox= obj.find("bndbox")
            x=float(bndbox.find("cx").text)
            y=float(bndbox.find("cy").text)
            w=float(bndbox.find("w").text)
            h=float(bndbox.find("h").text)
            a=float(bndbox.find("angle").text)
            objs.append([x,y,w,h,a])
        return objs

    def _draw_box(self,box,img):
        def obb2poly_np(bboxes):
            """convert box coordinate to corners

            Args:
                box (np.array): (5) x, y, w, h, alpha

            Returns:
                box (np.array): (4) xmin,ymin,xmax,ymax
            """
            x = bboxes[0:1]
            y = bboxes[1:2]
            w = bboxes[2:3]
            h = bboxes[3:4]
            alpha = bboxes[4:5] # (N, 1)
            x4 = np.array([0.5, -0.5, -0.5, 0.5])
            x4 = w * x4     # (B, N, 4)
            y4 = np.array([0.5, 0.5, -0.5, -0.5])
            y4 = h * y4      # (B, N, 4)
            corners = np.stack([x4, y4],axis=1)     # (N, 4, 2)
            sin = np.sin(alpha)
            cos = np.cos(alpha)
            row1 = np.concatenate([cos, sin], axis=-1)
            row2 = np.concatenate([-sin, cos], axis=-1)       # (N, 2)
            rot_T = np.stack([row1, row2], axis=-2)   # (N, 2, 2)
            rotated= np.einsum("ij,jk->ik",corners,rot_T)
            rotated[:,0] += x
            rotated[:,1] += y   #(N,4,2)
            return rotated
        c = obb2poly_np(box)
        c= c.reshape(8)
        corners_x =c[0::2]
        corners_y =c[1::2]
        cv2.line(img,(int(corners_x[0]),int(corners_y[0])),(int(corners_x[1]),int(corners_y[1])),[0,0,255],1)
        cv2.line(img,(int(corners_x[1]),int(corners_y[1])),(int(corners_x[2]),int(corners_y[2])),[0,0,255],1)
        cv2.line(img,(int(corners_x[2]),int(corners_y[2])),(int(corners_x[3]),int(corners_y[3])),[0,0,255],1)
        cv2.line(img,(int(corners_x[0]),int(corners_y[0])),(int(corners_x[3]),int(corners_y[3])),[0,0,255],1)
        cx =int(box[0])
        cy =int(box[1])
        v_y=int(0.5*box[3]*math.cos(-box[4]))
        v_x=int(0.5*box[3]*math.sin(-box[4]))
        cv2.line(img,(cx,cy),(cx-v_x,cy-v_y),[0,0,255],2)


    def _show(self):
        if self.img_path[-3:]== "jpg" or self.img_path[-3:]== "png":
            img=cv2.imread(self.img_path)
            assert os.path.isfile(self.xml_path)
            objs=self.load_xml_o(self.xml_path)
            for i in objs:
                self._draw_box(i,img)
            cv2.namedWindow("aaa",cv2.WINDOW_NORMAL)
            cv2.imshow("aaa",img)
            cv2.waitKey(0)
        else :
            for file in os.listdir(self.xml_path):
                img=cv2.imread(os.path.join(self.img_path,file[:-3]+"jpg"))
                out=os.path.join(self.root,"sample")
                if not os.path.isdir(out):
                    os.mkdir(out)
                # assert os.path.isfile(os.path.join(self.xml_path,file[:-3]+"xml"))
                objs=self.load_xml_o(os.path.join(self.xml_path,file))
                for i in objs:
                    self._draw_box(i,img)
                cv2.imwrite(os.path.join(out,file[:-3]+"jpg"),img)
                           

a=vis("/home/rvl224/圖片/box640/VOC")
a._show()