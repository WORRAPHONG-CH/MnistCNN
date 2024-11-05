# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:50:08 2022

@author: Worra
"""

import cv2
import numpy as np
import keras  
import matplotlib.pyplot as plt
from keras import datasets
from keras import layers
from keras.datasets import mnist
from keras.utils import np_utils
import random as rd
import os 

SEED = 32 #fixed random seed for training

os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)#random value set
rd.seed(SEED)

#%% Prepare The Data 

#Model and Data Parameters
num_classes = 10 
input_shape = (28,28,1)#(w,h,1 layer(gray?))

#Spilt data to train and test set
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#Normalize to scale 0-1
x_train = x_train.astype("float32")/255 
x_test = x_test.astype("float32")/255

#%% Make sure that image have shape(28,28,1)
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)
print("x_train shape: ",x_train.shape)#60000 image
print("x_test shape: ",x_test.shape)#10000 image 

#Convert class vector to binary matrices 
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)
print("y_train shape: ",y_train.shape)
print("y_test shape: ",y_test.shape)

#%% Build the model CNN 
model = keras.Sequential(
    [
     keras.Input(shape=input_shape),
     layers.Conv2D(16,kernel_size=(3,3),activation="relu"),
     layers.MaxPool2D(pool_size=(2,2)),
     layers.Conv2D(32,kernel_size=(3,3),activation="relu"),
     layers.MaxPool2D(pool_size=(2,2)),
     layers.Flatten(),#2D->1D
     layers.Dropout(0.5),
     layers.Dense(num_classes,activation="softmax"),#for multiclass
     ]
    )
model.summary()

#%% Train the model 
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

hist = model.fit(x_train,y_train,batch_size=64,epochs=5,validation_split=0.05)

#%% Evaulate the training model 
score = model.evaluate(x_test,y_test,verbose=1)
print("Test loss: ",score[0])
print("Test Accuracy: ",score[1]*100)

#%% Plot training graph
plt.figure("Graph Training",figsize=(9,6))
plt.subplot(121)
plt.title("Train Accuracy")
plt.plot(hist.history['accuracy'],label="Train Accuracy")
plt.xlabel("No. epoch")
plt.ylabel("Accuracy")

plt.subplot(122)
plt.title("Train Loss")
plt.plot(hist.history["loss"],label="Train Loss")
plt.xlabel("No. epoch")
plt.ylabel("Loss")
plt.show()

#%% Save model 
model.save("Model_CNN.h5")

#%% Test (prediction)
img_test = x_test[1000]
image_testRe = img_test.reshape((1,)+input_shape)
plt.imshow(img_test,cmap="gray")
predict = model.predict(image_testRe)
print("Prediction probability")
count = 0
prob_list = []

for i in predict.squeeze():
    i_percent = np.round(i*100,decimals=2)
    print(count,":",i_percent,"%")
    count +=1
    prob_list.append(i_percent)

prob_max = np.max(prob_list)
index_ans = prob_list.index(prob_max)
print("Answer: ",index_ans)
print("Probability Number: ",prob_max,"%")
