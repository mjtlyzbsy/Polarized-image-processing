#coding:utf-8
import numpy as np
from  matplotlib import pyplot as plt
import get_glcm
from PIL import  Image

def glcm_show(k, file):
    nbit = 64 #gray levels
    mi, ma = 0, 255 #max gray and min gray
    slide_window = 7
    #step = [2, 4, 8, 16]步长
    #angle = [0，45, 90, 135]角度
    step = [2]
    angle = [k*np.pi/4]

    img_ori = Image.open(file)
    img = np.array(img_ori.convert('L'))#如果图像有很多通道，则转为灰度图
    img = np.uint8(255.0 * (img - np.min(img))/(np.max(img) - np.min(img)))#归一化
    h, w = img.shape
    
    glcm = get_glcm.calcu_glcm(img, mi, ma, nbit, slide_window, step, angle)

    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            glcm_cut = np.zeros((nbit, nbit, h, w),dtype = np.float32)
            glcm_cut = glcm[:,:,i,j,:,:]
            glcm_mean = get_glcm.calcu_glcm_mean(glcm_cut, nbit)
            glcm_variance = get_glcm.calcu_glcm_variance(glcm_cut, nbit)
            glcm_energy = get_glcm.calcu_glcm_energy(glcm_cut,  nbit)
            glcm_contrast = get_glcm.calcu_glcm_contrast(glcm_cut,  nbit)
            glcm_correlation = get_glcm.calcu_glcm_correlation(glcm_cut,  nbit)
            glcm_entropy = get_glcm.calcu_glcm_entropy(glcm_cut, nbit)

    plt.figure(num = "{}°的偏振图像".format(k*45), figsize = (10, 4.5))
    font = {'family':'Times New Roman', 'size':12, }

    plt.subplot(2, 4, 1)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(img_ori, cmap='gray')
    plt.title('Original', font)

    plt.subplot(2, 4, 2)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.title('Gray', font)

    plt.subplot(2, 4, 3)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_mean, cmap='gray')
    plt.title('Mean', font)

    plt.subplot(2, 4, 4)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_variance, cmap='gray')
    plt.title('Variance', font)

    plt.subplot(2, 4, 5)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_energy, cmap='gray')
    plt.title('Angular Second Moment(Energy)', font)

    plt.subplot(2, 4, 6)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_contrast, cmap='gray')
    plt.title('Contrast', font)

    plt.subplot(2, 4, 7)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_correlation, cmap='gray')
    plt.title('Correlation', font)

    plt.subplot(2, 4, 8)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_entropy, cmap='gray')
    plt.title('Entropy', font)

    plt.tight_layout(pad=0.5)
    plt.show()