# -*- coding: utf-8 -*-
"""
# @Date    : 2020/5/20
# @Author  : Aqua
# @Email   : fy9390218@foxmail.com
# @File   : LSB-resize.py
# @Software: PyCharm
# @Desc    :
"""
from PIL import Image
import os
inpath = './new_obj'
outpath = './thumbnail'
items = os.listdir(inpath)
newlist = []
for names in items:
  if names.endswith(".jpg"):
      path = os.path.join(inpath, names)
      image = Image.open(path)
      image.thumbnail((416, 416))
      image.save('./thumbnail/{}'.format(names))

