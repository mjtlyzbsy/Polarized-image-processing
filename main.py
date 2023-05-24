import glcm_main
from multiprocessing import Process

if __name__ == '__main__':
    file = r"C:\Users\马骏涛\Desktop\毕业论文\实验图片\\"
    # file = input("请输入目标文件地址：")

    pic_show = []
    for i in range(4):
        image = file + "{}.png".format(i*45)
        pic_show.append(Process(target=glcm_main.glcm_show, args=(i, image)))
    # 启动线程
    for i in range(pic_show.__len__()):
        pic_show[i].start()
    for i in range(pic_show.__len__()):
        pic_show[i].join()
    glcm_main.glcm_show()