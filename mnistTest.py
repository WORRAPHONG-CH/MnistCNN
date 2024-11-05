# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 18:22:45 2022

@author: Worra
"""

import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from keras.datasets import mnist
from keras.models import load_model

model = load_model("C:/KMUTNB/Project_own/ImgProcessing/Model_CNN.h5")
model.summary()
model.get_weights()

#%% Input image
img = cv2.imread("./imgTest/9.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(img,cmap = "gray")
input_image = (28,28,1)

print(img.shape)
#normalize scale(0-1)
img = img/255.0
#%%
img = np.expand_dims(img,-1)

#%% Predict 
pred = model.predict(img.reshape((1,)+input_image))
count = 0
prob_list = []

for i in pred.squeeze():
    i_percent = np.round(i*100,decimals=2)
    print(count,":",i_percent,"%")
    count +=1
    prob_list.append(i_percent)

prob_max = np.max(prob_list)
index_ans = prob_list.index(prob_max)
print("Answer: ",index_ans)
print("Probability Number: ",prob_max,"%") 