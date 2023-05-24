import numpy as np
import cv2
from math import floor
from skimage.feature import graycomatrix


def image_patch(img2, slide_window, h, w):
    image = img2
    window_size = slide_window
    patch = np.zeros((slide_window, slide_window, h, w), dtype=np.uint8)

    for i in range(patch.shape[2]):
        for j in range(patch.shape[3]):
            patch[:, :, i, j] = img2[i : i + slide_window, j : j + slide_window]
 
    return patch

def calcu_glcm(img, vmin=0, vmax=255, nbit=64, slide_window=5, step=[2], angle=[0]):
    mi, ma = vmin, vmax
    h, w = img.shape

    # Compressed gray range：vmin: 0-->0, vmax: 256-1 -->nbit-1
    bins = np.linspace(mi, ma+1, nbit+1)
    img1 = np.digitize(img, bins) - 1

    # (512, 512) --> (slide_window, slide_window, 512, 512)
    img2 = cv2.copyMakeBorder(img1, floor(slide_window/2), floor(slide_window/2)
                              , floor(slide_window/2), floor(slide_window/2), cv2.BORDER_REPLICATE) # 图像扩充

    patch = np.zeros((slide_window, slide_window, h, w), dtype=np.uint8)
    patch = image_patch(img2, slide_window, h, w)

    # Calculate GLCM (5, 5, 512, 512) --> (64, 64, 512, 512)
    # greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
    glcm = np.zeros((nbit, nbit, len(step), len(angle), h, w), dtype=np.uint8)
    for i in range(patch.shape[2]):
        for j in range(patch.shape[3]):
            glcm[:, :, :, :, i, j]= graycomatrix(patch[:, :, i, j], step, angle, levels=nbit)
            # skimage.feature.graycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
            # image：输入的灰度图像。
            # distances：一个整数列表，表示要计算的邻距离。例如，[1, 2, 3]表示计算相邻像素之间的灰度共生矩阵以及相隔两个像素和三个像素的灰度共生矩阵。
            # angles：一个整数列表，表示要计算的角度（方向）。例如，[0, np.pi/4, np.pi/2, 3*np.pi/4]表示计算水平、对角线和垂直方向上的灰度共生矩阵。
            # levels：一个整数，表示图像的灰度级数目。如果未指定，将根据图像的最大灰度值自动确定。
            # symmetric：一个布尔值，指定是否考虑对称灰度对。如果为True，则在计算灰度共生矩阵时，考虑i和j之间的对称关系。默认为False。
            # normed：一个布尔值，指定是否对灰度共生矩阵进行标准化。如果为True，则对灰度共生矩阵进行标准化处理。默认为False。
            # 返回值： P[i,j,d,theta] 是灰度 j 在距离 d 处和与灰度 i 成角度 theta 处出现的次数
    return glcm

def calcu_glcm_mean(glcm, nbit=64):
# GLCM的均值特征

    mean = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i,j] * i / (nbit)**2

    return mean

def calcu_glcm_variance(glcm, nbit=64):
# GLCM的方差特征
    mean = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit)**2

    variance = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            variance += glcm[i, j] * (i - mean)**2

    return variance

def calcu_glcm_contrast(glcm, nbit=64):
# GLCM的对比度特征
    contrast = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            contrast += glcm[i, j] * (i-j)**2

    return contrast

def calcu_glcm_entropy(glcm, nbit=64):
# GLCM的熵特征
    eps = 0.00001
    entropy = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            entropy -= glcm[i, j] * np.log10(glcm[i, j] + eps)

    return entropy

def calcu_glcm_energy(glcm, nbit=64):
# GLCM的能量特征
    energy = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            energy += glcm[i, j]**2

    return energy

def calcu_glcm_correlation(glcm, nbit=64):
# GLCM的相关性特征
    mean = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit)**2

    variance = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            variance += glcm[i, j] * (i - mean)**2

    correlation = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    variance[np.where(variance == 0.0)] = 0.001#防止除以0的情况
    for i in range(nbit):
        for j in range(nbit):
            correlation += np.float32((i - mean) * (j - mean) * (glcm[i, j]**2)) / variance

    return correlation