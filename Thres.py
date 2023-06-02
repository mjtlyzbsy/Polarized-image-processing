import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import io
from PIL import Image

def thresh(file):
    img = np.array(Image.open(file).convert('L'))

    _, thresh1 = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh1, thresh2

def main(path):  
    # 读取图像文件
    file = path + '\\Itot.png'
    thresh1, thresh2 = thresh(file)
    io.imsave(path + '\\Itot_Thres.png', thresh1)
    io.imsave(path + '\\Itot_OTSU.png', thresh2)
    
    file = path + '\\AoLP.png'
    thresh1, thresh2 = thresh(file)
    io.imsave(path + '\\AoLP_Thres.png', thresh1)
    io.imsave(path + '\\AoLP_OTSU.png', thresh2)

    file = path + '\\DoLP.png'
    thresh1, thresh2 = thresh(file)
    io.imsave(path + '\\DoLP_Thres.png', thresh1)
    io.imsave(path + '\\DoLP_OTSU.png', thresh2)

if __name__ == '__main__':
    path = r'E:\code\_code\png\mirror'
    main(path)