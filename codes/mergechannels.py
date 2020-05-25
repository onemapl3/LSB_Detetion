# -*- coding: utf-8 -*-
"""
# @Date    : 2020/5/18
# @Author  : Aqua
# @Email   : fy9390218@foxmail.com
# @File   : LSB-mergechannels.py
# @Software: PyCharm
# @Desc    :
"""
from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
import math
import scipy.misc
from PIL import Image, ImageChops

def img_normalization(img):
    '''
    图像归一化
    '''
    img=img.astype(np.float32)
    img[np.isnan(img)] = 0
    if np.amax(img) == 0:
        return img
    else:
        img -= np.amin(img)
        img = img/(np.amax(img)-np.amin(img))
        img *= 255
        return img.astype(np.uint8)

def limit(image_data):


    mean = np.mean(image_data)
    std = np.std(image_data)
    max = mean+std
    min = mean-std
    image_data[image_data>max] = max
    image_data[image_data<min] = min
    return image_data

csv_data = pd.read_csv("./target.csv")
labelimg_g = csv_data.loc[:,['name_g']].values
labelimg_r = csv_data.loc[:,['name_r']]
rootdir = './target'
list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
num=0
for value in labelimg_g:
    for i in range(0, len(list)):

        if list[i] == value:
            path1 = os.path.join(rootdir, list[i])
            path2 = os.path.join(rootdir, list[i][:11]+'r'+list[i][12:])
            image1 = fits.getdata(path1, cache=True)
            image2 = fits.getdata(path2, cache=True)
            image1 = limit(image1)
            image2 = limit(image2)
            img1 = img_normalization(image1)
            img2 = img_normalization(image2)
            b = np.zeros((img1.shape[0], img1.shape[1]), dtype=img1.dtype)
            g = np.zeros((img1.shape[0], img1.shape[1]), dtype=img1.dtype)
            r = np.zeros((img2.shape[0], img2.shape[1]), dtype=img2.dtype)
            #
            g[:, :] = img1[:, :]
            r[:, :] = img2[:, :]
            #
            merged = cv2.merge([b, g, r])
            cv2.imwrite("./new_img/{}.jpg".format(num),merged)
            num+=1
            break

# path1 = './target/fpC-001035-g2-0011.fit'
# path2 = './target/fpC-001035-r2-0011.fit'


# image1 = fits.getdata(path1, cache=True)
# image2 = fits.getdata(path2, cache=True)

# img=Image.fromarray(np.uint8(image1/float(math.pow(2,16)-1)*255))
# img.save("./1.png")
# img1 = img_normalization(image1)
# img2 = img_normalization(image2)
# [rows, cols] = img.shape
# for i in range(rows):
#  for j in range(cols):
#   if img[i,j]==255:
#       print(i,j)
# cv2.imwrite("./1.png",img)

# im1 = Image.fromarray(image1)
# im2 = Image.fromarray(image2)
# im1.save("./labeled_img/fpC-001035-g2-0011.png")
# im2.save("./labeled_img/fpC-001035-r2-0011.png")
# plt.savefig("./labeled_img/fpC-001035-g2-0011.png")
# plt.imshow(image2,cmap="gray")
# plt.savefig("./labeled_img/fpC-001035-r2-0011.png")

# img = cv2.imread("./labeled_img/fpC-001035-g2-0011.png",0)
# img2 = cv2.imread("./labeled_img/fpC-001035-r2-0011.png",0)
# b = np.zeros((img1.shape[0], img1.shape[1]), dtype=img1.dtype)
# g = np.zeros((img1.shape[0], img1.shape[1]), dtype=img1.dtype)
# r = np.zeros((img2.shape[0], img2.shape[1]), dtype=img2.dtype)
# #
# g[:, :] = img1[:, :]
# r[:, :] = img2[:, :]
# #
# merged = cv2.merge([b, g, r])
# cv2.imwrite("./merged.png",merged)
# cv2.imshow("Merged", merged)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# img_1 = cv2.imread("./labeled_img/fpC-001035-g2-0011.png")
# img_2 = cv2.imread("./labeled_img/fpC-001035-r2-0011.png")
# mul1 = ImageChops.add(image1, image2, scale=2)
# mul1.show()
# needed_multi_channel_img = np.zeros((1489,2048, 3))
# needed_multi_channel_img [:,:,0] = image1
# needed_multi_channel_img [:,:,1] = image2
#
# #
# cv2.imwrite("./needed_multi_channel_img.png",needed_multi_channel_img)

# no_img = 2
# img = img_1/no_img + img_2/no_img
# cv2.imshow("merged",img)