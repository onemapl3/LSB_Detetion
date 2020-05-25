# -*- coding: utf-8 -*-
"""
# @Date    : 2020/5/18
# @Author  : Aqua
# @Email   : fy9390218@foxmail.com
# @File   : LSB-find_multi.py
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
name = df['name_g'].values
d={}
l=[]
for i,v in enumerate(name):
    if  v not in d:
        d[v]=i
    else:
        print(d[v],i,v)


