# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:14:53 2019

@author: cyuan
"""

from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K


# 训练和测试的图片分为'bus', 'dinosaur', 'flower', 'horse', 'elephant'五类
# 其图片的下载地址为 http://pan.baidu.com/s/1nuqlTnN ,总共500张图片,其中图片以3,4,5,6,7开头进行按类区分
# 训练图片400张，测试图片100张；注意下载后，在train和test目录下分别建立上述的五类子目录，keras会按照子目录进行分类识别
NUM_CLASSES = 5
TRAIN_PATH = r'C:\Users\cyuan\Desktop\ML_summary\keras迁移学习--图片识别\train'
TEST_PATH = r'C:\Users\cyuan\Desktop\ML_summary\keras迁移学习--图片识别\test'
# 代码最后挑出一张图片进行预测识别
PREDICT_IMG = r'C:\Users\cyuan\Desktop\ML_summary\keras迁移学习--图片识别\test\elephant\502.jpg'
# FC层定义输入层的大小
FC_NUMS = 1024
# 冻结训练的层数，根据模型的不同，层数也不一样，根据调试的结果，VGG19和VGG16c层比较符合理想的测试结果，
#本文采用VGG19做示例
FREEZE_LAYERS = 17
# 进行训练和测试的图片大小，VGG19推荐为224×244
IMAGE_SIZE = 224

# 采用VGG19为基本模型，include_top为False，表示FC层是可自定义的，
# 抛弃模型中的FC层；该模型会在~/.keras/models下载基本模型
base_model = VGG19(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')

# 自定义FC层以基本模型的输入为卷积层的最后一层
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(FC_NUMS, activation='relu')(x)
prediction = Dense(NUM_CLASSES, activation='softmax')(x)

# 构造完新的FC层，加入custom层
model = Model(inputs=base_model.input, outputs=prediction)
# 可观察模型结构
model.summary()
# 获取模型的层数
print("layer nums:", len(model.layers))


# 除了FC层，靠近FC层的一部分卷积层可参与参数训练，
# 一般来说，模型结构已经标明一个卷积块包含的层数，
# 在这里我们选择FREEZE_LAYERS为17，表示最后一个卷积块和FC层要参与参数训练
for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model.layers[FREEZE_LAYERS:]:
    layer.trainable = True
for layer in model.layers:
    print("layer.trainable:", layer.trainable)

# 预编译模型
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# 给出训练图片的生成器， 其中classes定义后，可让model按照这个顺序进行识别
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(directory=TRAIN_PATH,
                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE), classes=['bus', 'dinosaur', 'flower', 'horse', 'elephant'])
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(directory=TEST_PATH,
                                                  target_size=(IMAGE_SIZE, IMAGE_SIZE), classes=['bus', 'dinosaur', 'flower', 'horse', 'elephant'])

# 运行模型
model.fit_generator(train_generator, epochs=5, validation_data=test_generator)


# 找一张图片进行预测验证
img = load_img(path=PREDICT_IMG, target_size=(IMAGE_SIZE, IMAGE_SIZE))
# 转换成numpy数组
x = img_to_array(img)
# 转换后的数组为3维数组(224,224,3),
# 而训练的数组为4维(图片数量, 224,224, 3),所以我们可扩充下维度
x = K.expand_dims(x, axis=0)
# 需要被预处理下
x = preprocess_input(x)
# 数据预测
result = model.predict(x, steps=1)
# 最后的结果是一个含有5个数的一维数组，我们取最大值所在的索引号，即对应'bus', 'dinosaur', 'flower', 'horse', 'elephant'的顺序
print("result:", K.eval(K.argmax(result)))