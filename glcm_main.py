#coding:utf-8
import numpy as np
from  matplotlib import pyplot as plt
import get_glcm
from PIL import  Image

def glcm_show(path, file):
    nbit = 8 #gray levels
    mi, ma = 0, 255 #max gray and min gray
    slide_window = 4
    step = [1]
    angle = [0,np.pi/4.0,np.pi/2.0,np.pi/4.0*3.0]

    file = path + '\\' + file
    img = np.array(Image.open(file).convert('L'))#如果图像有很多通道，则转为灰度图
    img = np.uint8(255.0 * (img - np.min(img))/(np.max(img) - np.min(img)))#归一化
    h, w = img.shape
    
    glcm = get_glcm.calcu_glcm(img, mi, ma, nbit, slide_window, step, angle)

    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            glcm_cut = np.zeros((nbit, nbit, h, w),dtype = np.float32)
            glcm_cut = glcm[:,:,i,j,:,:]
            glcm_energy = get_glcm.calcu_glcm_energy(glcm_cut,  nbit)
            glcm_contrast = get_glcm.calcu_glcm_contrast(glcm_cut,  nbit)
            glcm_correlation = get_glcm.calcu_glcm_correlation(glcm_cut,  nbit)
            glcm_entropy = get_glcm.calcu_glcm_entropy(glcm_cut, nbit)

    plt.figure(num = "灰度共生矩阵")
    font = {'family':'SimSun', 'size':12, }

    plt.subplot(2, 2, 1)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_energy, cmap='gray')
    plt.title('Angular Second Moment(Energy)', font)

    plt.subplot(2, 2, 2)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_contrast, cmap='gray')
    plt.title('Contrast', font)

    plt.subplot(2, 2, 3)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_correlation, cmap='gray')
    plt.title('Correlation', font)

    plt.subplot(2, 2, 4)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(glcm_entropy, cmap='gray')
    plt.title('Entropy', font)

    plt.tight_layout(pad=0.5)
    plt.savefig(path + '\glcm.png')
    plt.show(block = 'False')
    
    print(  "  ASM:{}".format(np.mean(glcm_energy))
          , "  CON:{}".format(np.mean(glcm_contrast))
          , "  CORRLN:{}".format(np.mean(glcm_correlation))
          , "  ENT:{}".format(np.mean(glcm_entropy)))

    glcm_energy = np.uint8(255.0 * (glcm_energy - np.min(glcm_energy))/(np.max(glcm_energy) - np.min(glcm_energy)))
    glcm_energy = Image.fromarray(glcm_energy, mode='L')
    glcm_energy.save(path + '\\glcm_ASM.png', quality = 95)
    glcm_contrast = np.uint8(255.0 * (glcm_contrast - np.min(glcm_contrast))/(np.max(glcm_contrast) - np.min(glcm_contrast)))
    glcm_contrast = Image.fromarray(glcm_contrast, mode='L')
    glcm_contrast.save(path + '\\glcm_CON.png', quality = 95)
    glcm_correlation = np.uint8(255.0 * (glcm_correlation - np.min(glcm_correlation))/(np.max(glcm_correlation) - np.min(glcm_correlation)))
    glcm_correlation = Image.fromarray(glcm_correlation, mode='L')
    glcm_correlation.save(path + '\\glcm_CORRLN.png', quality = 95)
    glcm_entropy = np.uint8(255.0 * (glcm_entropy - np.min(glcm_entropy))/(np.max(glcm_entropy) - np.min(glcm_entropy)))
    glcm_entropy = Image.fromarray(glcm_entropy, mode='L')
    glcm_entropy.save(path + '\\glcm_ENT.png', quality = 95)