import numpy as np 
import os
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore',category = FutureWarning)
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Conv2D , MaxPooling2D , Activation , Flatten


data_dir = '/home/andrei/Desktop/Tensorflow/cats_vs_dogs/kagglecatsanddogs_3367a/PetImages'
categories = ['Dog','Cat']
IMG_SIZE = 80

# for category in categories:
# 	path = os.path.join(data_dir,category)
# 	for img in os.listdir(path):
# 		img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
# 		new_img = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE) )
# 		#plt.imshow(new_img,cmap = 'gray')
# 		#plt.show() 
# 		break

training_data = []

def create_training_data():
	for category in categories:
		path = os.path.join(data_dir,category)
		cls_label = categories.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
				new_img = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE) )
				training_data.append([new_img,cls_label])
			except Exception as e:
				pass
			

# create_training_data()

# random.shuffle(training_data)

# X = []
# y = []
# for feature,label in training_data:
# 	X.append(feature)
# 	y.append(label)
# X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
# y = np.array(y)

# np.save('training_data.npy', X)
# np.save('labels.npy',y)
X = np.load('training_data.npy')
y = np.load('labels.npy')

model = Sequential()

model.add(Conv2D(64 , (3,3) , input_shape = X.shape[1:] ))
model.add(Activation('relu'))
model.add(MaxPooling2D( pool_size = (2,2)))

model.add(Conv2D(64 , (3,3) ))
model.add(Activation('relu'))
model.add(MaxPooling2D( pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(X,y, epochs = 3 ,validation_split = 0.1 , batch_size = 32)

