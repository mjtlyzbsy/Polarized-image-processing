import numpy as np
from skimage.io import imread
from tqdm import trange
from PIL import  Image

def FCM(file, clusters, max_iter=100, fuzziness=4.0, epsilon=1e-5):
    image=np.array(Image.open(file).convert('L'), dtype=np.float64)

    # 初始化隶属度矩阵
    rows, cols = image.shape
    membership = np.random.rand(rows, cols, clusters)
    membership /= np.sum(membership, axis=2, keepdims=True, dtype=np.float64)# 每一簇的μ和都是1
    centers = np.zeros((clusters,))

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
                distances = np.abs(image[i][j] - centers) #该点到所有聚类中心的价值距离
                # distances = np.linalg.norm(image[i][j] - centers)
                membership_new[i][j] = 1.0 / np.power(distances, 2.0 / (fuzziness - 1))
                membership_new[i][j] /= np.sum(membership_new[i][j])
        
        # 判断是否收敛
        diff = np.sum(np.abs(membership - membership_new), dtype=np.float64)
        if diff < epsilon:
            break
        else:
            membership = membership_new
    
    centers = np.uint8(centers)
    # print(np.uint8(centers))
    # 根据隶属度生成聚类图像
    cluster_image = np.argmax(membership, axis=2)
    cluster_image = centers[cluster_image]

    return cluster_image

def main(path, clusters, fuzziness):
    # 读取图像文件
    file = path + '\\Itot.png'
    cluster_image = FCM(file, clusters, fuzziness)
    img_Itot = Image.fromarray(cluster_image, mode='L')
    img_Itot.save(path + '\\Itot_FCM.png', quality = 95)
    
    file = path + '\\AoLP.png'
    cluster_image = FCM(file, clusters, fuzziness)
    img_AoLP = Image.fromarray(cluster_image, mode='L')
    img_AoLP.save(path + '\\AoLP_FCM.png', quality = 95)

    file = path + '\\DoLP.png'
    cluster_image = FCM(file, clusters, fuzziness)
    img_DoLP = Image.fromarray(cluster_image, mode='L')
    img_DoLP.save(path + '\\DoLP_FCM.png', quality = 95)

if __name__ == '__main__':
    path = r'E:\code\_code\png\mirror'
    clusters = 5
    fuzziness = 10
    main(path, clusters, fuzziness)