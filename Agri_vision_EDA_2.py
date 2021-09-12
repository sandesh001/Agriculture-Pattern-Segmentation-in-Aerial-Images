# -*- coding: utf-8 -*-
"""
Created on Sun May 16 11:55:09 2021

@author: Sandesh
"""
import numpy as np

import os 
import cv2
from tqdm import tqdm 
import matplotlib.pyplot as plt

p = "F:\\PhD\\Implementation\\MY_Segmentation\\Agriculture-Vision\\train\\labels\\"
classes = os.listdir(p)

path = 'F:\\PhD\\Implementation\\MY_Segmentation\\Agriculture-Vision\\train\\labels\\cloud_shadow\\'
ids = os.listdir(path)

cl=[]
bl=[]


for i in ids:
    img = cv2.imread(path + i)
    img = np.array(img)
    #print(len(np.unique(img)))
    if len(np.unique(img))==2:
        cl.append(i)
    else:
        bl.append(i)
                




for c in classes:
    for i in ids:
        img = cv2.imread(p + c +"\\"+ i)
        img = np.array(img)
        #print(len(np.unique(img)))
        if len(np.unique(img))==2:
            cl.append(i)
        else:
            bl.append(i)
                
    print('For ' + c+' class:', len(cl)/12901)
    print(len(cl),len(bl))
    cl=[]
    bl=[]
    

classes 
################################

p = "F:\\PhD\\Implementation\\MY_Segmentation\\Agriculture-Vision\\train\\labels\\"
classes = os.listdir(p)

rgb_path = "F:\\PhD\\Implementation\\MY_Segmentation\\Agriculture-Vision\\train\\images\\rgb\\"
rgb_img = os.listdir(rgb_path)

count =[]

for c in classes:
    cl_img = os.listdir(p + c+'\\') 
    for cl in cl_img:
        for rgb in rgb_img: 
            if cl[:-4]==rgb[:-4]:
                count.append(cl)
                img = cv2.imread(rgb_path + rgb )
                img = np.array(img)
                plt.imsave('New_rgb/'+rgb,img)
                
#############################################################################               
        img = cv2.imread(p + c +"\\"+ i)
        img = np.array(img)
        #print(len(np.unique(img)))
        if len(np.unique(img))==2:
            cl.append(i)
            plt.imsave('New_data/'+c+'/'+i,img)

#save classwise images to folder
for c in classes:
    for i in ids:
        img = cv2.imread(p + c +"\\"+ i)
        img = np.array(img)
        #print(len(np.unique(img)))
        if len(np.unique(img))==2:
            cl.append(i)
            plt.imsave('New_data/'+c+'/'+i,img)
        else:
            bl.append(i)
                
    print('For ' + c+' class:', len(cl)/12901)
    print(len(cl),len(bl))
    cl=[]
    bl=[]