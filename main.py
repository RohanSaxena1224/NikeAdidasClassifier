#Connecting Google Drive files to Google Colab
from google.colab import drive
drive.mount('/content/drive')

#Importing Libraries that will be needed 
from PIL import Image
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt

#Making a file path for test/train data
TRAIN_DIR = './drive/My Drive/NIKE ADDIDAS/TRAIN'
TEST_DIR ='./drive/My Drive/NIKE ADDIDAS/TEST'

#Setting constants so that each image is the same
IMG_SIZE = 120
LR = 1e-3


#One hot encoding Nike and Adidas images
#Nike is [1,0] and Adidas [0,1]
def label_img(img):
  #Images are formatted as: ADIDAS_x, NIKE_x
  word_label = img.split('_')[0]
  if word_label == 'NIKE' : 
    return[1,0]
  elif word_label == 'ADIDAS':
     return[0,1]

    
#Preprocessing of training data
def create_train_data():
  train_data = []

  #TQDM shows how long the function is taking by displaying a progress bar
  for imgName in tqdm(os.listdir(TRAIN_DIR)):
    label = label_img(imgName)
    #Path to a specific image as a concatenation
    path = os.path.join(TRAIN_DIR, imgName)
   
    img = Image.open(path)
    img = img.convert('L') #Convert it to greyscale
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

    train_data.append([np.array(img), np.array(label)])

  shuffle(train_data)
  #Safety measure incase the program takes a long time
  np.save('./drive/My Drive/NIKE ADDIDAS/TRAIN.npy', train_data) 
  return train_data
  create_train_data()

# Similar code can be used for the testing data


train_data = create_train_data()
plt.imshow(train_data[43][0], cmap = 'gray')
#Test cell, making sure the images display properly


#Processing test data function
def process_test_data():
  test_data = []

  for img in tqdm(os.listdir(TEST_DIR)):
    path = os.path.join(TEST_DIR, img)
    if "DS_Store" not in path:
      img_num = img.split('_')[1]

      img = Image.open(path)
      img = img.convert('L')
      img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

      test_data.append([np.array(img), img_num])
  
  shuffle(test_data)
  return test_data


train_data = create_train_data()
plt.imshow(train_data[43][0], cmap = 'gist_gray')
print(train_data[43][1])
#Testing again


!pip install tensorflow==1.4
!pip install tflearn
#Installation of tensorflow since it will be used


#packages for tensorflow needed to build network
import tflearn
import tflearn.layers.conv 
from tflearn.layers.conv import conv_2d, max_pool_2d

import tflearn.layers.core 
from tflearn.layers.core import input_data, dropout, fully_connected

import tflearn.layers.estimator 
from tflearn.layers.estimator import regression 

import tensorflow as tf
tf.reset_default_graph()

#Convolutional Neural Network Architecture
convnet = input_data(shape=[None, IMG_SIZE,IMG_SIZE, 1], name = 'input')
convnet = conv_2d(convnet, 32, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64,5,activation='relu')
convnet = max_pool_2d(convnet,5)
convnet = conv_2d(convnet, 32,5,activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64,5,activation='relu')
convnet = max_pool_2d(convnet,5)
convnet = conv_2d(convnet, 32,5,activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64,5,activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024,activation='relu')
convnet = dropout(convnet,0.8)
convnet = fully_connected(convnet,2,activation='softmax')
convnet = regression(convnet, optimizer = 'adam', learning_rate = LR, loss = 'categorical_crossentropy', name = 'targets')
model = tflearn.DNN(convnet,tensorboard_verbose=3)

train = train_data[-90:]
test = train_data[:-90]


#shaping data to fit for training/testing
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

#The model
model.fit({'input':X},{'targets':Y},n_epoch=100,validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=50,show_metric=True, run_id='NIKE ADDIDAS')

#Showing how well the model was able to predict the data
test_data = process_test_data()
fig = plt.figure()

for num, data in enumerate(test_data[:10]):
  img_num = data[1]
  img_data = data[0]

  y = fig.add_subplot(3,4,num+1)
  orig = img_data
  data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
  model_out = model.predict([data])[0]
  print(model_out)
  if np.argmax(model_out) == 1:
    str_label= 'ADIDAS'
  else: 
    str_label = 'NIKE'

  y.imshow(orig, cmap = 'gray')
  plt.title(str_label)
  y.axes.get_xaxis().set_visible(False)
  y.axes.get_yaxis().set_visible(False)

plt.show()







    
