#!/usr/bin/env python

# coding: utf-8

# In[1]:

import os
from albumentations import *
import cv2
from tensorflow.keras.utils import Sequence
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from tensorflow.image import convert_image_dtype
import tensorflow as tf

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class BatchGenerator(Sequence):
    def __init__(self, train_file, phase, batch_size, class_list, net_h, net_w, shuffle = True):
        self.train_file = train_file
        self.phase = phase
        self.batch_size = batch_size
        self.class_list = class_list
        self.shuffle = shuffle      
        self.net_h = net_h
        self.net_w = net_w
        
        if self.shuffle :
            np.random.shuffle(self.train_file)
    
    def load_img(self, path):
        return plt.imread(path)
    
    def __len__(self):
        return math.ceil(len(self.train_file) / self.batch_size)
    
    def preprocess_img(self, img):
        return cv2.resize(img, (self.net_w, self.net_h))

    def on_epoch_end(self):
        if self.shuffle :
            np.random.shuffle(self.train_file)   
   
    def __getitem__(self, idx):     
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size
        
        if r_bound > len(self.train_file):
            r_bound = len(self.train_file)
            l_bound = r_bound - len(self.train_file)
        
        x = []
        y = []
        for ind, i in enumerate(range(l_bound, r_bound)):   
            if ind >= self.batch_size or ind < 0:
                continue
            img_name = self.train_file[i]
            nir = self.load_img('dataset/'+self.phase+'/images/nir/'+img_name+'.jpg')
            rgb = self.load_img('dataset/'+self.phase+'/images/rgb/'+img_name+'.jpg')
            combine = self.preprocess_img(rgb)
            #combine = self.preprocess_img(np.concatenate([rgb, np.expand_dims(nir, -1)], axis = -1))
            x.append(combine)
            
            if self.phase == 'test':
                continue
            images = []
            for class_name in class_list:
                img = self.load_img('dataset/'+self.phase+'/labels/'+class_name+'/'+img_name+'.png')
                images.append(np.expand_dims(img, -1))
            label = self.preprocess_img(np.concatenate(images, axis = -1))
            
            y.append(label)
        
        if self.phase == 'test':
            return np.array(x)
        return np.array(x), np.array(y)


# In[2]:


import numpy as np
import os
from sklearn.model_selection import train_test_split

batch_size = 16
net_h = 512
net_w = 512

train_file = np.load('train.npy')
val_file = np.load('val.npy')

#train_file, val_file = train_test_split(train_file, test_size = 0.1, random_state = 0)

class_list = ['cloud_shadow', 'double_plant', 'planter_skip', 'standing_water', 'waterway', 'weed_cluster']
#class_list = ['cloud_shadow', 'double_plant', 'planter_skip', 'standing_water', 'waterway']


train_generator = BatchGenerator(train_file, 'train', batch_size, class_list, net_h, net_w)
val_generator = BatchGenerator(val_file, 'val', batch_size, class_list, net_h, net_w)
#val_generator = BatchGenerator(test_file, 'val', batch_size, class_list, net_h, net_w)

#print(train_generator[0].shape)
# In[3]:


smooth = 1e-15
def mean_iou(y_true, y_pred):
    y_pred = tf.round(tf.cast(y_pred, tf.int32))
    intersect = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32), axis=[1])
    union = tf.reduce_sum(tf.cast(y_true, tf.float32),axis=[1]) + tf.reduce_sum(tf.cast(y_pred, tf.float32),axis=[1])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

def iou_loss(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection 
    score = (intersection + smooth) /(union + smooth) 
    return 1 - score


def dice_coeff(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +
                                           smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg +
                                  (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)
# In[4]:

smooth = 1e-15
def dice_coeff(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss
    
def bce_dice_loss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


smooth = 1e-15
def dice_coefff(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


from losses import (
    binary_crossentropy,
    dice_loss,
    bce_dice_loss,
    dice_coef,
    weighted_bce_dice_loss
)

from tensorflow.keras.callbacks import *
#from tensorflow_addons.losses import GIoULoss
#from tensorflow.keras.metrics import MeanIoU
#from models import unet_res
from DeepLabV3_model import Deeplabv3

#from resnext_fpn import resnext_fpn

#model = resnext_fpn((net_h, net_w, 4),5)

model = Deeplabv3(input_shape=(net_h, net_w, 3), activation='softmax',classes=7)

#model = unet_res(num_classes = 5, input_shape = (net_h, net_w, 4))
model.compile(loss = weighted_bce_dice_loss,
             optimizer = 'adam',
             metrics = [mean_iou, dice_coefff])
callbacks = [
    CSVLogger('csv_logs/DeepLabV3_focal_loss.csv'),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint('weights/DeeplabV3_weighted_bce_dice_loss.h5', verbose=1, save_best_only=True), 
    EarlyStopping(patience = 5, verbose = 1)
]

model.summary()

model.load_weights('DeeplabV3_weighted_bce_dice_loss.h5')


# In[ ]:
'''
model.fit(x = train_generator,
          steps_per_epoch = train_file.shape[0]//batch_size,
          epochs = 200,
          validation_data = val_generator, 
          validation_steps = val_file.shape[0]//batch_size,
          callbacks = callbacks, 
          verbose = 1)
'''
# In[ ]:

#model.load_weights('weights/DeeplabV3_weighted_bce_dice_loss.h5')
#model.evaluate(test_generator, verbose = 1)
from PIL import Image
from tqdm import tqdm


# Val Images
#val_img_dir = '/home/sandesh/Desktop/Agri_Vision_2021/supervised/Agriculture-Vision-2021/test/images/rgb/'
val_img_dir= '/home/sandesh/Desktop/Agriculture Vision/dataset/test/images/rgb/'
test_nir = '/home/sandesh/Desktop/Agriculture Vision/dataset/test/images//nir/'

x = (os.listdir(val_img_dir))
y = (os.listdir(test_nir))

#for read and save
for i in tqdm(range(len(x))):
    test_rgb = plt.imread(val_img_dir + x[i])
    val_img = np.array(test_rgb,dtype=np.float32)
    pred = model.predict(np.expand_dims(val_img,axis=0))
    mask = np.argmax(pred[0],axis=-1)
    #mask = convert_dtype(mask)
    #print(mask.shape)
    #plt.imsave('Pred_2021_img/'+x[i][:-4]+'.png',mask)
    #mask.save('Prediction1/'+x[p][:-4]+'.png')
    result = Image.fromarray((mask).astype(np.uint8))
    result.save('Pred_W2021_S20200/'+x[i][:-4]+'.png')


'''
X = []
for i in range(len(x)):
    test_rgb = plt.imread(val_img_dir + x[i])
    #test_nir = plt.imread(test_nir + y[i])
    #img = np.concatenate([test_rgb, np.expand_dims(test_nir, -1)], axis = -1) 
    #img = cv2.resize(img,(224,224), interpolation = cv2.INTER_AREA)
    #img = cv2.cvtColor(test_rgb, cv2.COLOR_BGR2GRGB)
    X.append(test_rgb)

val_img = np.array(X,dtype=np.float32)
print(val_img.shape)
print(val_img[0].shape)
from PIL import Image
from tqdm import tqdm

def convert_dtype(img):
    f= np.zeros(shape = (img.shape[0], img.shape[1]), dtype = np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            f[i][j] = img[i][j]
    
    return f

for p in tqdm(range(len(val_img))):
    pred = model.predict(np.expand_dims(val_img[p],axis=0))
    mask = np.argmax(pred[0],axis=-1)
    #mask = convert_dtype(mask)
    #print(mask.shape)
    #plt.imsave('Prediction/'+x[p][:-4]+'.png',mask)
    #mask.save('Prediction1/'+x[p][:-4]+'.png')
    result = Image.fromarray((mask).astype(np.uint8))
    result.save('Pred_2021/'+x[p][:-4]+'.png')
'''
'''


# Val Images
rgb_path = '/mnt/komal/Sandesh/Agriculture _Vision/Agriculture Vision/dataset/test/images/rgb/'
nir_path = 'D:/Machine Learning/Agriculture Vision/Data/test/images/nir/'

from PIL import Image
from tqdm import tqdm

write_folder = 'submission/'
os.mkdir(write_folder)
def write_image(im_arr, im_name='test'):
    Image.fromarray(np.uint8(im_arr), 'L').save(os.path.join(write_folder, im_name))

for img_name in tqdm(os.listdir(rgb_path)):
    img = plt.imread(rgb_path + img_name)
    out_probs = model(tf.expand_dims(img, 0))[0]
    out_cat = np.argmax(out_probs, axis=-1)
    write_image(out_cat, img_name)
'''
