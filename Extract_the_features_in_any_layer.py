# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:26:03 2019

@author: cyuan
"""

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from keras.utils import plot_model
from matplotlib import pyplot as plt

#【0】VGG19模型，加载预训练权重
base_model = VGG19(weights='imagenet')

#【1】创建一个新model, 使得它的输出(outputs)是 VGG19 中任意层的输出(output)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
print(model.summary())                                 # 打印模型概况
plot_model(model,to_file = 'a simple convnet.png')     # 画出模型结构图，并保存成图片




#【2】从网上下载一张图片，保存在当前路径下
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224)) # 加载图片并resize成224x224

#【3】将图片转化为4d tensor形式
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

#【4】数据预处理
x = preprocess_input(x) #去均值中心化，preprocess_input函数详细功能见注释
"""
def preprocess_input(x, data_format=None, mode='caffe'):
   Preprocesses a tensor or Numpy array encoding a batch of images.

    # Arguments
        x: Input Numpy or symbolic tensor, 3D or 4D.
        data_format: Data format of the image tensor/array.
        mode: One of "caffe", "tf".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.

    # Returns
        Preprocessed tensor or Numpy array.

    # Raises
        ValueError: In case of unknown `data_format` argument.
"""
#【5】提取特征
block4_pool_features = model.predict(x)
print(block4_pool_features.shape) #(1, 14, 14, 512)