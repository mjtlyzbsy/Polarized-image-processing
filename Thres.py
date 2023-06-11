import cv2
import numpy as np
from skimage import io
from PIL import Image
from  matplotlib import pyplot as plt

def thresh(file):
    img = np.array(Image.open(file).convert('L'))

    _, thresh1 = cv2.threshold(img,40,255,cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh1, thresh2

def main(path):  
    # 读取图像文件
    file = path + '\\Itot.png'
    thresh1, thresh2 = thresh(file)
    io.imsave(path + '\\Itot_Thres.png', thresh1)
    io.imsave(path + '\\Itot_OTSU.png', thresh2)
    
    file = path + '\\AoLP.png'
    thresh3, thresh4 = thresh(file)
    io.imsave(path + '\\AoLP_Thres.png', thresh3)
    io.imsave(path + '\\AoLP_OTSU.png', thresh4)

    file = path + '\\DoLP.png'
    thresh5, thresh6 = thresh(file)
    io.imsave(path + '\\DoLP_Thres.png', thresh5)
    io.imsave(path + '\\DoLP_OTSU.png', thresh6)
    
    plt.figure(num = "阈值分割")
    font = {'family':'SimSun', 'size':12, }
    
    plt.subplot(2, 3, 1)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(thresh1, cmap='gray')
    plt.title('强度图像典型阈值分割', font)

    plt.subplot(2, 3, 4)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(thresh2, cmap='gray')
    plt.title('强度图像Otus算法', font)

    plt.subplot(2, 3, 2)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(thresh3, cmap='gray')
    plt.title('AoLP图像典型阈值分割', font)

    plt.subplot(2, 3, 5)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(thresh4, cmap='gray')
    plt.title('AoLP图像Otus算法', font)

    plt.subplot(2, 3, 3)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(thresh5, cmap='gray')
    plt.title('DoLP图像典型阈值分割', font)

    plt.subplot(2, 3, 6)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(thresh6, cmap='gray')
    plt.title('DoLP图像Otus算法', font)

    plt.tight_layout(pad=0.5)
    plt.savefig(path + '\Thres.png')
    plt.show(block = 'False')