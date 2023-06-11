import AoLp_DoLP
import Thres
import FCM
import glcm_main
import vectorfields

if __name__ == '__main__':
    str = input('输入目标文件地址:')
    if(str == ''):
        path = r'..\png'
    else:
        path = str
    
    print(path)

    print("偏振角图像 偏振度图像 强度图像")
    AoLp_DoLP.main(path)

    print("偏振方向矢量场")
    vectorfields.VF(path)

    print("偏振角图像纹理特性分析")
    glcm_main.glcm_show(path, 'AoLP.png')
    
    print("阈值分割算法")
    Thres.main(path)
    
    print("FCM图像分割算法")
    FCM.main(path)