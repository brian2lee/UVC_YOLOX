# def data_aug():



import cv2
import numpy as np
 
# 讀取第一幅圖像
image = cv2.imread('/home/rvl224/文件/wilbur_data/VOC/JPEGImages/416.jpg')
 
import cv2
import numpy as np
 
def rotate_image(image, angle):
    # 取得圖像的高度和寬度
    (h, w) = image.shape[:2]
    # 計算圖像的中心點
    center = (w // 2, h // 2)
    # 取得旋轉矩陣
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 旋轉圖像
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image
 
# 示例: 將圖像旋轉 90 度
rotated_image = rotate_image(image, 90)
 
# 顯示結果
cv2.imshow('image', image)
cv2.imshow('rotated_image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()