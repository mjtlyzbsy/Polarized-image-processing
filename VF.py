import numpy as np
from PIL import  Image
import matplotlib.pyplot as plt

def VF(path):
    # 读取图像
    file1 = path+'\\0.png'
    file2 = path+'\\45.png'
    file3 = path+'\\90.png'
    file4 = path+'\\135.png'
    I0 = np.array(Image.open(file1).convert('L'), dtype=np.float64)
    I45 = np.array(Image.open(file2).convert('L'), dtype=np.float64)
    I90 = np.array(Image.open(file3).convert('L'), dtype=np.float64)
    I135 = np.array(Image.open(file4).convert('L'), dtype=np.float64)
    
    I = 0.5 * (I0 + I45 + I90 + I135)
    Q = I0 - I90
    U = I45 - I135

    I[np.where(I == 0.0)] = 0.0001
    Q[np.where(Q == 0.0)] = 0.0001
    
    Itot = I
    DOLP = np.sqrt(Q**2 + U**2) / I
    AOLP = np.float64(0.5 * np.arctan(U, Q) * 180.0 / np.pi)
    AOLP = (AOLP + 180) % 180
    I_disp = Itot / 2

    x, y = I_disp.shape
    x = np.int16(np.linspace(0, x-1, x))
    y = np.int16(np.linspace(0, y-1, y))
    x, y = np.meshgrid(x, y)
    u = DOLP * np.cos(2 * AOLP * np.pi / 180)     # 求出波的水平分量
    v = DOLP * np.sin(2 * AOLP * np.pi / 180)     # 求出波的垂直分量

    plt.imshow(I_disp, cmap='gray')
    plt.quiver(x, y, u, v, color = 'y', scale = 2, width =0.002, headwidth = 2, pivot = 'middle')
    plt.show()

def main(path):
    VF(path)

if __name__ == '__main__':
    path = r'E:\code\_code\png\mirror'
    main(path)