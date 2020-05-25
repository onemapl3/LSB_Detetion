# -*- coding: utf-8 -*-
"""
# @Date    : 2020/5/23
# @Author  : Aqua
# @Email   : fy9390218@foxmail.com
# @File   : LSB-mergenew.py
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
labelimg_r = csv_data.loc[:,['name_r']].values
rootdir = './target'
list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
num=0

d={}
for i in range(0, len(list)):
    if (list[i] in labelimg_g) or (list[i] in labelimg_r):
        print(list[i])
        print(i)
        continue
    else:
        if list[i] not in d:
            if list[i][11] == 'g':
                path1 = os.path.join(rootdir, list[i])
                path2 = os.path.join(rootdir, list[i][:11]+'r'+list[i][12:])
            else:
                path1 = os.path.join(rootdir, list[i])
                path2 = os.path.join(rootdir, list[i][:11] + 'g' + list[i][12:])
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
            cv2.imwrite("./dj/{}.jpg".format(list[i][:11]+'gr'+list[i][12:]),merged)
            d[list[i][:11]+'g'+list[i][12:]]=1
            d[list[i][:11]+'r'+list[i][12:]]=1


