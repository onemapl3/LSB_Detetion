# -*- coding: utf-8 -*-
"""
# @Date    : 2020/5/19
# @Author  : Aqua
# @Email   : fy9390218@foxmail.com
# @File   : LSB-generate_train.py
# @Software: PyCharm
# @Desc    :
"""
import os

image_files = []
os.chdir(os.path.join("data", "obj"))
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("data/obj/" + filename)
os.chdir("..")
with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")