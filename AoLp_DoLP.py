import numpy as np
from PIL import  Image

def AoLp_DoLP(path):
    # 读取图像
    file1 = path + '\\0.png'
    file2 = path + '\\45.png'
    file3 = path + '\\90.png'
    file4 = path + '\\135.png'
    I0 = np.array(Image.open(file1).convert('L'), dtype=np.float64)
    I45 = np.array(Image.open(file2).convert('L'), dtype=np.float64)
    I90 = np.array(Image.open(file3).convert('L'), dtype=np.float64)
    I135 = np.array(Image.open(file4).convert('L'), dtype=np.float64)

    I = 0.5*(I0 + I45 + I90 + I135)
    Q = I0 - I90
    U = I45 - I135

    I[np.where(I == 0.0)] = 0.0001
    Q[np.where(Q == 0.0)] = 0.0001

    Itot = np.uint8(0.5 * I)
    AoLP = np.int8(0.5 * np.arctan(U / Q)*180 / np.pi)
    AoLP = np.uint8((AoLP + 180) % 180)
    DoLP = np.uint8(np.sqrt(pow(Q, 2) + pow(U, 2)) / I*256.0)

    img_Itot = Image.fromarray(Itot, mode='L')
    img_Itot.save(path + '\\Itot.png', quality = 95)

    img_AoLP = Image.fromarray(AoLP, mode='L')
    img_AoLP.save(path + '\\AoLP.png', quality = 95)

    img_DoLP = Image.fromarray(DoLP, mode='L')
    img_DoLP.save(path + '\\DoLP.png', quality = 95)

def main(path):
    AoLp_DoLP(path)

if __name__ == '__main__':
    path = r'E:\code\_code\png\mirror'
    main(path)