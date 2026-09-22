
from cmath import pi
import cv2 
import numpy as np
import xml.etree.ElementTree as ET
import math
import torch
def obb2hbb_np(bboxes):
    """convert box coordinate to corners

    Args:
        box (np.array): ( N, 5) with x, y, w, h, alpha

    Returns:
        box (np.array): (N, 4) xmin,ymin,xmax,ymax
    """
    x = bboxes[:, 0:1]
    y = bboxes[:, 1:2]
    alpha = bboxes[:, 4:5] # (N, 1)
    w = bboxes[:, 2:3]
    h = bboxes[:, 3:4]
    
    # alpha = (2*math.pi*np.ones((bboxes.shape[0],1)))%math.pi
    x4 = np.array([0.5, -0.5, -0.5, 0.5])
    x4 = w * x4     # (B, N, 4)
    y4 = np.array([0.5, 0.5, -0.5, -0.5])
    y4 = h * y4      # (B, N, 4)
    corners = np.stack([x4, y4],axis=2)     # (N, 4, 2)
    sin = np.sin(alpha)
    cos = np.cos(alpha)
    row1 = np.concatenate([cos, sin], axis=-1)
    row2 = np.concatenate([-sin, cos], axis=-1)       # (N, 2)
    rot_T = np.stack([row1, row2], axis=-2)   # (N, 2, 2)
    rotated= np.einsum("ijk,ikn->ijn",corners,rot_T)
    rotated[..., 0] += x
    rotated[..., 1] += y   #(N,4,2)
    l =np.min(rotated[:,:,0],axis=1)
    t =np.min(rotated[:,:,1],axis=1)
    r =np.max(rotated[:,:,0],axis=1)
    b =np.max(rotated[:,:,1],axis=1)
    hbb =np.stack([l,t,r,b],axis=1)   # (N, 4) 
    return rotated,hbb

def xml2ls(path):
    target = ET.parse(path).getroot()
    res = np.empty((0, 5))
    for obj in target.iter("object"):
        bndbox = []
        bbox = obj.find("bndbox")
        pts = ["cx","cy","w","h"] 
        for i, pt in enumerate(pts):
            cur_pt = int(float(bbox.find(pt).text)) - 1   
            bndbox.append(cur_pt)
        _theta = round(float(bbox.find("angle").text),5)
        bndbox.append(_theta)
        res = np.vstack((res, bndbox))
    return res

a = torch.tensor([])
print(len(a) == 0)

M=np.array([[ 1.17297569e+00, -2.16676725e-02,  3.14265574e+01],
       [-3.26166754e-02,  1.17297569e+00,  1.80241146e+01]])
res=xml2ls("/home/rvl224/文件/wilbur_data/VOC/Annotations_o/164.xml")
c,hdd=obb2hbb_np(res)
ch=c.reshape(len(res)*4,2)
img =cv2.imread("/home/rvl224/文件/wilbur_data/VOC/JPEGImages/164.jpg")
corner_points = np.ones((4 * len(res), 3))
corner_points[:,:2]=ch
corner_points = corner_points @ M.T
corner_points =corner_points.reshape(len(res),8)
# aa= np.argmax(corner_points,axis=1)
img = cv2.warpAffine(img, M, dsize=(410,280), borderValue=(114, 114, 114))
for i in range(res.shape[0]):
        corners=corner_points[i]
        corners_x =corners[0::2]
        corners_y =corners[1::2]
        a=corners_x.max()
        

        if corners_x.min()>0 and corners_x.min()<410 and corners_y.min()>0 and corners_y.max()<280:
            cv2.line(img,(int(corners_x[0]),int(corners_y[0])),(int(corners_x[1]),int(corners_y[1])),[0,0,255],1)
            cv2.line(img,(int(corners_x[1]),int(corners_y[1])),(int(corners_x[2]),int(corners_y[2])),[0,0,255],1)
            # cv2.line(img,(int(corners_x[2]),int(corners_y[2])),(int(corners_x[3]),int(corners_y[3])),[0,0,255],1)
            # cv2.line(img,(int(corners_x[0]),int(corners_y[0])),(int(corners_x[3]),int(corners_y[3])),[0,0,255],1)

print(np.concatenate([a,b],axis=1))
cv2.imshow("aa",img)
cv2.waitKey(0)

