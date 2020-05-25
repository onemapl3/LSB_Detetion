# -*- coding: utf-8 -*-
"""
# @Date    : 2020/5/18
# @Author  : Aqua
# @Email   : fy9390218@foxmail.com
# @File   : LSB-format.py
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
df = pd.read_csv("./target.csv")
x=df['X'].values
y=df['Y'].values



for i in range(1129):
    img = Image.open("./objv3/{}.jpg".format(i))

    img = np.array(img)
    x[i] =round(x[i]/img.shape[1],6)
    y[i] = round(y[i]/img.shape[0],6)

    if x[i]>1 or y[i]> 1:
        print(i,img.shape[0],img.shape[1])

    with open('./new_txt/{}.txt'.format(i), "w+") as f:
        f.writelines("0 {} {} {} {}".format(x[i],y[i],0.050000,0.050000))




