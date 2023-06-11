import numpy as np
from tqdm import trange
from PIL import  Image, ImageOps
from  matplotlib import pyplot as plt

def FCM(file, clusters, max_iter=100, fuzziness=20.0, epsilon=1e-5):
    image=np.array(Image.open(file).convert('L'), dtype=np.float64)

    # 初始化隶属度矩阵
    rows, cols = image.shape
    membership = np.random.dirichlet(np.ones(clusters), size=(rows, cols))# 每一簇的μ和都是1
    centers = np.random.rand(clusters)*255

    # 迭代更新隶属度和聚类中心
    for _ in trange(max_iter):
        # 计算聚类中心
        for k in range(clusters):
            centers[k] = np.sum(np.power(membership[:, :, k], fuzziness) * image, dtype=np.float64) / np.sum(np.power(membership[:, :, k], fuzziness), dtype=np.float64)
        # 更新隶属度
        membership_new = np.zeros_like(membership)
        distances = np.zeros(clusters)
        for i in range(rows):
            for j in range(cols):
                distances = np.abs(image[i][j] - centers) 
                membership_new[i][j] = 1.0 / np.power(distances, 2.0 / (fuzziness - 1))
                membership_new[i][j] /= np.sum(membership_new[i][j])
        
        # 判断是否收敛
        diff = np.sum(np.abs(membership - membership_new), dtype=np.float64)
        if diff < epsilon:
            break
        else:
            membership = membership_new
    
    # 根据隶属度生成聚类图像
    cluster_image = np.argmax(membership, axis=2)
    maxx = np.max(centers)
    minn = np.min(centers)
    centers = np.uint8((centers - minn) * 255.0/(maxx - minn))
    cluster_image = centers[cluster_image]

    return cluster_image

def main(path, clusters = 4, max_iter = 30):
    # 读取图像文件
    print('强度图像FCM算法分割')
    file = path + '\\Itot.png'
    cluster_image = FCM(file, clusters, max_iter)
    img_Itot = Image.fromarray(cluster_image, mode='L')
    img_Itot = ImageOps.equalize(img_Itot)
    img_Itot.save(path + '\\Itot_FCM.png', quality = 95)
    
    print('AoLP图像FCM算法分割')
    file = path + '\\AoLP.png'
    cluster_image = FCM(file, clusters, max_iter)
    img_AoLP = Image.fromarray(cluster_image, mode='L')
    img_AoLP = ImageOps.equalize(img_AoLP)
    img_AoLP.save(path + '\\AoLP_FCM.png', quality = 95)

    print('DoLP图像FCM算法分割')
    file = path + '\\DoLP.png'
    cluster_image = FCM(file, clusters, max_iter)
    img_DoLP = Image.fromarray(cluster_image, mode='L')
    img_DoLP = ImageOps.equalize(img_DoLP)
    img_DoLP.save(path + '\\DoLP_FCM.png', quality = 95) 

    plt.figure(num = "FCM算法")
    font = {'family':'SimSun', 'size':12, }

    plt.subplot(1, 3, 1)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(img_Itot, cmap='gray')
    plt.title('强度图像FCM算法分割', font)

    plt.subplot(1, 3, 2)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(img_AoLP, cmap='gray')
    plt.title('偏振角图像FCM算法分割', font)

    plt.subplot(1, 3, 3)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(img_DoLP, cmap='gray')
    plt.title('偏振度图像FCM算法分割', font)
    plt.savefig(path + '\FCM.png')
    plt.show(block = 'False')