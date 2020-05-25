# -*- coding: utf-8 -*-
"""
# @Date    : 2020/5/20
# @Author  : Aqua
# @Email   : fy9390218@foxmail.com
# @File   : LSB-img_process.py
# @Software: PyCharm
# @Desc    :
"""
import cv2
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from numpy import histogram, interp
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

image_data = fits.getdata('./target/fpC-001035-g2-0011.fit')
# plt.imshow(image_data, cmap='gray')
# plt.colorbar()




def histeq(img, nbr_bins=65536):
    """ Histogram equalization of a grayscale image. """
    # 获取直方图p(r)
    imhist, bins = histogram(img.flatten(), nbr_bins, normed = True)

    # 获取T(r)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 65535 * cdf / cdf[-1]

    # 获取s，并用s替换原始图像对应的灰度值
    result = interp(img.flatten(), bins[:-1], cdf)

    return result.reshape(img.shape), cdf

def equalization16bit(img):


    flat = img.flatten()
    hist,bins = histogram(img.flatten(), 65536)
    # plt.plot(hist)
    #
    cs = hist.cumsum()
    # re-normalize cumsum values to be between 0-255

    # numerator & denomenator
    nj = (cs - cs.min()) * 65535
    N = cs.max() - cs.min()

    # re-normalize the cdf
    cs = nj / N
    cs = cs.astype('uint16')
    img_new = cs[flat]
    # plt.hist(img_new, bins=65536)
    # plt.show(block=True)
    img_new = np.reshape(img_new, img.shape)
    return img_new
    # cv2.imwrite("contrast.jpg", img_new)

# img,cdf = histeq(image_data)
#first method
img = equalization16bit(image_data)
plt.hist(img.flatten(), bins="auto")

plt.show()

# img_normalization(img)
cv2.imshow("img",img)
cv2.imwrite("./contrast1.tif",img)
# img8 = (image_data/256).astype('uint8')
#second method
# mean = np.mean(image_data)
# std = np.std(image_data)
# max = mean+std
# min = mean-std
# image_data[image_data>max] = max
# image_data[image_data<min] = min

# img = equalization16bit(image_data)
# img_normalization(img)
# cv2.imshow(img)
# img = img_normalization(image_data)

# hist = cv2.calcHist([img],[0],None,[256],[0,255])
# plt.plot(hist,'r')plt.hist(img.flatten(), 65536)


#

#
#
#
# plt.show()
#
# cv2.imshow("result", img)
# cv2.imwrite("./new.jpg",img)
# img = img_normalization(img)
# plt.imshow(img,cmap='gray')
# plt.show()
print('Min:', np.min(image_data))
print('Max:', np.max(image_data))
print('Mean:', np.mean(image_data))
print('Stdev:', np.std(image_data))
#
# out = hist_equal(image_data,np.min(image_data),np.max(image_data))
#
# cv2.imshow("result", out)

#
# dst = cv2.equalizeHist(image_data)

# plt.hist(image_data.flatten(), bins='auto')
# img=image_data.copy()
# Maximg = np.max(img)
# Minimg = np.min(img)
# Omin, Omax = 0, 255
# a = float(Omax - Omin) / (Maximg - Minimg)
# b = Omin - a * Minimg
# print(a, b, '-----------')
# # 线性变换
# O = a * img + b
# O = O.astype(np.uint8)
# # 利用灰度直方图进行比较  mget为GrayHist中的写方法
# print('Min:', np.min(O))
# print('Max:', np.max(O))
# print('Mean:', np.mean(O))
# print('Stdev:', np.std(O))
# plt.hist(O.flatten(), bins='auto')
# plt.show()
# # cv2.imshow('enhance', O)
cv2.waitKey(0)
cv2.destroyAllWindows()