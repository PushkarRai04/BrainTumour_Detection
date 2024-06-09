import cv2
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
image_directory='archive/'
no_tumor=os.listdir(image_directory+ 'no/')
yes_tumor=os.listdir(image_directory+ 'yes/')
dataset=[]
label=[]
INPUT_SIZE=64
for i, img_name in enumerate(no_tumor):
    if(img_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+ 'no/' +img_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)
for i, img_name in enumerate(yes_tumor):
    if(img_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+ 'yes/' +img_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)
print(len(label))
dataset=np.array(dataset)
label=np.array(label)
x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0 )
#reshape = (n, image width, image height, n channe)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

#Model building

model=Sequential()
model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model=Sequential()
model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model=Sequential()
model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
# Binary CrossEntropy =1, sigmoid
# Cross Entropy=2, softmax

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train,
batch_size=16,
verbose=1, epochs=10,
validation_data=(x_test, y_test),
shuffle=False)
model.save('BrainTumor10Epochs.h5')