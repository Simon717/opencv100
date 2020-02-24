import numpy as np
import cv2
import os

# 定义常量
dir = r"D:\learn\ImageProcessing100Wen-master\Question_01_10"
dir_img = os.path.join(dir, "answers_image")

def cvshow(img, name="img"):
    cv2.namedWindow(name, flags=0)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

