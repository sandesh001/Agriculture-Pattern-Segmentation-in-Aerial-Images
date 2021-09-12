# -*- coding: utf-8 -*-
"""
Created on Sun May 16 11:55:09 2021

@author: Sandesh
"""
import numpy as np

import os 
import cv2
import tqdm 

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