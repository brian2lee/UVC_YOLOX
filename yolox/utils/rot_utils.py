import math
from tokenize import Double
import torch
import numpy as np
from cuda_op.cuda_ext import sort_v
EPSILON = 1e-8
__all__ = [
    "cal_iou",
    "decode_box",
    "poly2hdd",
    "obb2hbb_np",
    "obb2poly_np",
    "box2corners_th",
    "theta2vector",
    "vector2theta",
    "aoI_select"

]

def theta2vector(angs):
    '''
    input torch.Tensor : (N,1) ->theta
    return torch.Tensor : (N,2) ->vector(x,y)
    '''
    dx = torch.sin(angs[...,0]).unsqueeze(1)
    dy = torch.cos(angs[...,0]).unsqueeze(1)
    return torch.cat((dx,dy),dim=-1)

def vector2theta(v):
    '''
    input torch.Tensor : (N,2) ->vector(x,y)
    return torch.Tensor : (N,1) ->theta
    '''
    vx = v[...,0]
    vy = v[...,1]
    vx = torch.where(torch.abs(vx)<1e-3,torch.full_like(vx,1e-3),vx)
    vy = torch.where(torch.abs(vy)<1e-3,torch.full_like(vx,1e-3),vy)
    mask =vy <0.0
    t = -torch.arctan((vx/vy))
    t=(
        (math.pi*mask.float()-t)%(2*math.pi)
    ).unsqueeze(-1)
    return t

def poly2hdd(bboxes):
    """Args:
        corners1 (torch.Tensor): N, 4, 2
    Return:
        horizon box's top left & bottom right(torch.Tensor):N,4
    """
    l = torch.min(bboxes[...,0],-1).values
    t = torch.min(bboxes[...,1],-1).values
    r = torch.max(bboxes[...,0],-1).values
    b = torch.max(bboxes[...,1],-1).values
    return torch.cat((l.view(-1,1),t.view(-1,1)
                    ,r.view(-1,1),b.view(-1,1)),-1)
def obb2hbb_np(bboxes):
    """convert box coordinate to corners

    Args:
        box (np.array): ( N, 5) with x, y, w, h, alpha

    Returns:
        box (np.array): (N, 4) xmin,ymin,xmax,ymax
    """
    x = bboxes[:, 0:1]
    y = bboxes[:, 1:2]
    w = bboxes[:, 2:3]
    h = bboxes[:, 3:4]
    alpha = bboxes[:, 4:5] # (N, 1)
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
    return hbb

def obb2poly_np(bboxes):
    """convert box coordinate to corners

    Args:
        box (np.array): ( N, 5) with x, y, w, h, alpha

    Returns:
        box (np.array): (N, 4) xmin,ymin,xmax,ymax
    """
    x = bboxes[:, 0:1]
    y = bboxes[:, 1:2]
    w = bboxes[:, 2:3]
    h = bboxes[:, 3:4]
    alpha = bboxes[:, 4:5] # (N, 1)
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
    return rotated
#piou

def box_intersection_th(corners1:torch.Tensor, corners2:torch.Tensor):
    """find intersection points of rectangles
    Convention: if two edges are collinear, there is no intersection point

    Args:
        corners1 (torch.Tensor): number of ground truth ,number of predict , 4, 2
        corners2 (torch.Tensor): number of ground truth ,number of predict , 4, 2

    Returns:
        intersectons (torch.Tensor): number of ground truth ,number of predict , 4, 4, 2
        mask (torch.Tensor) : number of ground truth ,number of predict , 4, 4; bool
    """

    # build edges from corners
    line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3) # N, 4, 4: Box, edge, point
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)
    # duplicate data to pair each edges from the boxes
    # (B, N, 4, 4) -> (B, N, 4, 4, 4) : Batch, Box, edge1, edge2, point
    line1_ext = line1.unsqueeze(3).repeat([1,1,1,4,1])
    line2_ext = line2.unsqueeze(2).repeat([1,1,4,1,1])
    x1 = line1_ext[..., 0]
    y1 = line1_ext[..., 1]
    x2 = line1_ext[..., 2]
    y2 = line1_ext[..., 3]
    x3 = line2_ext[..., 0]
    y3 = line2_ext[..., 1]
    x4 = line2_ext[..., 2]
    y4 = line2_ext[..., 3]
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    num = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)     
    den_t = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
    t = den_t / num
    t[num == .0] = -1.
    mask_t = (t > 0) * (t < 1)                # intersection on line segment 1
    den_u = (x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)
    u = -den_u / num
    u[num == .0] = -1.
    mask_u = (u > 0) * (u < 1)                # intersection on line segment 2
    mask = mask_t * mask_u 
    t = den_t / (num + EPSILON)                 # overwrite with EPSILON. otherwise numerically unstable
    intersections = torch.stack([x1 + t*(x2-x1), y1 + t*(y2-y1)], dim=-1)
    intersections = intersections * mask.float().unsqueeze(-1)
    return intersections, mask


def box1_in_box2(corners1:torch.Tensor, corners2:torch.Tensor):
    """check if corners of box1 lie in box2
    Convention: if a corner is exactly on the edge of the other box, it's also a valid point

    Args:
        corners1 (torch.Tensor): (number of ground truth ,number of predict , 4, 2)
        corners2 (torch.Tensor): (number of ground truth ,number of predict , 4, 2)

    Returns:
        c1_in_2: (number of ground truth ,number of predict , 4) Bool
    """
    a = corners2[:, :, 0:1, :]  # (number of ground truth ,number of predict , 1, 2)
    b = corners2[:, :, 1:2, :]  # (number of ground truth ,number of predict , 1, 2)
    d = corners2[:, :, 3:4, :]  # (number of ground truth ,number of predict , 1, 2)
    ab = b - a                  # (number of ground truth ,number of predict ,1, 2)
    am = corners1 - a           # (number of ground truth ,number of predict , 4, 2)
    ad = d - a                  # (number of ground truth ,number of predict ,1, 2)
    p_ab = torch.sum(ab * am, dim=-1)       # (number of ground truth ,number of predict , 4)
    norm_ab = torch.sum(ab * ab, dim=-1)    # (number of ground truth ,number of predict , 1)
    p_ad = torch.sum(ad * am, dim=-1)       # (number of ground truth ,number of predict ,4)
    norm_ad = torch.sum(ad * ad, dim=-1)    # (number of ground truth ,number of predict , 1)
    # NOTE: the expression looks ugly but is stable if the two boxes are exactly the same
    # also stable with different scale of bboxes
    cond1 = (p_ab / norm_ab > - 1e-6) * (p_ab / norm_ab < 1 + 1e-6)   # (B, N, 4)
    cond2 = (p_ad / norm_ad > - 1e-6) * (p_ad / norm_ad < 1 + 1e-6)   # (B, N, 4)
    return cond1*cond2

def box_in_box_th(corners1:torch.Tensor, corners2:torch.Tensor):
    """check if corners of two boxes lie in each other

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        c1_in_2: (B, N, 4) Bool. i-th corner of box1 in box2
        c2_in_1: (B, N, 4) Bool. i-th corner of box2 in box1
    """
    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1)
    return c1_in_2, c2_in_1

def build_vertices(corners1:torch.Tensor, corners2:torch.Tensor, 
                c1_in_2:torch.Tensor, c2_in_1:torch.Tensor, 
                inters:torch.Tensor, mask_inter:torch.Tensor):
    """find vertices of intersection area

    Args:
        corners1 (torch.Tensor): ( N, 4, 2)
        corners2 (torch.Tensor): ( N, 4, 2)
        c1_in_2 (torch.Tensor): Bool, (N, 4)
        c2_in_1 (torch.Tensor): Bool, ( N, 4)
        inters (torch.Tensor): (N, 4, 4, 2)
        mask_inter (torch.Tensor): (N, 4, 4)
    
    Returns:
        vertices (torch.Tensor): (N, 24, 2) vertices of intersection area. only some elements are valid
        mask (torch.Tensor): (N, 24) indicates valid elements in vertices
    """
    # NOTE: inter has elements equals zero and has zeros gradient (masked by multiplying with 0). 
    # can be used as trick
    GT = corners1.size()[0]
    pred = corners1.size()[1]
    vertices = torch.cat([corners1, corners2, inters.view([GT,pred, -1, 2])], dim=2) # (B, N, 4+4+16, 2)
    mask = torch.cat([c1_in_2, c2_in_1, mask_inter.view([GT,pred, -1])], dim=2) # Bool (B, N, 4+4+16)
    return vertices, mask

def sort_indices(vertices:torch.Tensor, mask:torch.Tensor):
    """[summary]

    Args:
        vertices (torch.Tensor): float (N, 24, 2)
        mask (torch.Tensor): bool (N, 24)

    Returns:
        sorted_index: bool (N, 9)
    
    Note:
        why 9? the polygon has maximal 8 vertices. +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X) 
        and X indicates the index of arbitary elements in the last 16 (intersections not corners) with 
        value 0 and mask False. (cause they have zero value and zero gradient)
    """
    num_valid = torch.sum(mask.int(), dim=2).int()      # (N)
    mean = torch.sum(vertices * mask.float().unsqueeze(-1), dim=2, keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)
    vertices_normalized = vertices - mean      # normalization makes sorting easier
    return sort_v(vertices_normalized, mask, num_valid).long()

def calculate_area(idx_sorted:torch.Tensor, vertices:torch.Tensor):
    """calculate area of intersection

    Args:
        idx_sorted (torch.Tensor): (B, N, 9)
        vertices (torch.Tensor): (B, N, 24, 2)
    
    return:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding 
    """
    
    idx_ext = idx_sorted.unsqueeze(-1).repeat([1,1,1,2])
    selected = torch.gather(vertices, 2, idx_ext)
    total = selected[:, :, 0:-1, 0]*selected[:, :, 1:, 1] - selected[:, :, 0:-1, 1]*selected[:, :, 1:, 0]
    total = torch.sum(total, dim=2)
    area = torch.abs(total) / 2
    return area, selected

def oriented_box_intersection_2d(corners1:torch.Tensor, corners2:torch.Tensor):
    """calculate intersection area of 2d rectangles 

    Args:
        corners1 (torch.Tensor): (N, 4, 2)
        corners2 (torch.Tensor): (N, 4, 2)

    Returns:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding 
    """
    
    inters, mask_inter = box_intersection_th(corners1, corners2)
    c12, c21 = box_in_box_th(corners1, corners2)
    vertices, mask = build_vertices(corners1, corners2, c12, c21, inters, mask_inter)
    sorted_indices = sort_indices(vertices, mask)
    # sorted_indices = sorted_indices.view(mask.shape[0],mask.shape[1],mask.shape[2],9)
    return calculate_area(sorted_indices, vertices)



def decode_box(boxes):
    x = boxes[:,0]
    y = boxes[:,1]
    W = boxes[:,2]
    H = boxes[:,3]
    R1 = torch.clamp(boxes[:,4],0.001,0.999)
    R2 = torch.clamp(boxes[:,5],0.001,0.999)
    w1 = torch.sqrt(torch.pow(W*R1,2)+torch.pow(H*R2,2))
    w2 = torch.sqrt(torch.pow(W*(1-R1),2)+torch.pow(H*(1-R2),2))
    maxW = torch.maximum(w1,w2)
    tmp_v = w1==maxW
    T_tmp_v = w1 != maxW
    t1 = torch.full((1,boxes.shape[0]),0.5*math.pi,device="cuda:0")+torch.atan(-(H*R2)/(W*R1))
    t1= t1.squeeze(0)
    t2 = torch.atan((H*(1-R2))/(W*(1-R1)))
    t = tmp_v * t1 + T_tmp_v * t2
    out_boxes = torch.stack((x,y,w2,w1,t),dim=1)
    # a=torch.max(t)
    # b=torch.min(t)
    
    
    return out_boxes





def box2corners_th(box:torch.Tensor)-> torch.Tensor:
    """convert box coordinate to corners

    Args:
        box (torch.Tensor): (N, 5) with x, y, w, h, alpha

    Returns:
        torch.Tensor: (N, 4, 2) corners
    """
    B = box.size()[0]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5] # (B, N, 1)
    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).to(box.device) # (1,1,4)
    x4 = x4 * w     # (N, 4)
    y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).to(box.device)
    y4 = y4 * h     # (N, 4)
    corners = torch.stack([x4, y4], dim=-1)     # (N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)       # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)   # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1,4,2]), rot_T.view([-1,2,2]))
    # rotated = rotated.view([-1,4,2])          # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated

def cal_iou(box1:torch.Tensor, box2:torch.Tensor):
    """calculate iou
    
    Args:
        box1 (torch.Tensor): (N, 5)
        box2 (torch.Tensor): (N, 5)
    
    Returns:
        iou (torch.Tensor): (B, N)
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners1 (torch.Tensor): (B, N, 4, 2)
        U (torch.Tensor): (B, N) area1 + area2 - inter_area
    """
    if box1.shape[1]==6:
        box1 = decode_box(box1)
        box2 = decode_box(box2)
    # box1[:,0:2] = box1[:,0:2]+1000.0
    # box2[:,0:2] = box2[:,0:2]+1000.0
    box1 = box1.cuda()
    box2 = box2.cuda()

    corners1 = box2corners_th(box1)
    corners2 = box2corners_th(box2)
    num_gt = corners1.shape[0]
    num_pred = corners2.shape[0]
    corners1 = corners1.view(-1,1,4,2).expand(num_gt,num_pred,4,2)
    corners2 = corners2.expand(num_gt,num_pred,4,2)
    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)        #(B, N)
    area1 = box1[:, 2] * box1[:, 3]
    area2 = box2[:, 2] * box2[:, 3]
    area1 = area1.view(-1,1).expand(num_gt,num_pred)
    area2 = area2.expand(num_gt,num_pred)
    area1 = torch.clamp(area1,min=1e-8)
    area2 = torch.clamp(area2,min=1e-8)
    inter_area = torch.clamp(inter_area,1e-16)
    u = area1 + area2 - inter_area
    iou = inter_area / u
    iou = torch.nan_to_num(iou,nan=0.0)
    return iou

def aoI_select(box,img_size,threshold):
    boxes_Area = (box[2]-box[0])*(box[3]-box[1])
    img_Area = img_size[0]*img_size[1]
    Area_in_image = (min(img_size[0],box[2])-max(0,box[0]))*(min(img_size[1],box[3])-max(0,box[1]))
    aoI = Area_in_image / boxes_Area  
    if (box[2]+box[0])/2 < img_size[0] and (box[3]+box[1])/2 < img_size[1]:
        if aoI>=threshold:
            return False
        else :
            return True
    else :
        return True 