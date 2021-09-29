import tensorflow as tf

from keras.utils import np_utils 

from matplotlib import pyplot as plt 

import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

print('Training Images: {}'.format(X_train.shape))

print('Testing Images: {}'.format(X_test.shape))

print(X_train[0].shape)

for i in range (332,336):
  
  #plt. subplot(120+ 1 + i)
  
   plt.figure(figsize=(4,4)) 
   
   img = X_train[i]
   
   plt.imshow(img)
   
   plt.show()

X_train=X_train.reshape(X_train.shape[0], 32, 32, 3) 

X_test= X_test.reshape(X_test.shape[0], 32, 32, 3)

X_train= X_train.astype('float32')

X_test= X_test.astype('float32')

X_train /=255

X_test=X_test/255

n_classes = 10

print("shape before one-hot encoding: ", y_train.shape) 

y_train=np_utils.to_categorical (y_train, n_classes) 

y_test=np_utils.to_categorical(y_test, n_classes) 

print("shape after one-hot encoding: ", y_train.shape)

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Dropout,MaxPool2D, Flatten

model=Sequential()

#convolutional layers

model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))

model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')) 

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')) 

model.add(MaxPool2D(pool_size=(2, 2))) 

model.add(Dropout(0.25))

model.add(Flatten())

#hidden layer

model.add(Dense(500, activation='relu')) 

model.add(Dropout(0.4)) 

model.add(Dense(250, activation='relu'))

model.add(Dropout(0.3))

#output layer

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer="adam")

#training the model

model.fit(X_train, y_train, batch_size=128, epochs=10)

classes= range(0, 10)

names= ['airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog', 
        'horse', 
        'ship',
        'truck']

#zip the names and classes to make a dictionary of class_labels

class_labels = dict(zip(classes, names))

#generate batch of 9 images to predict 

batch =X_test[100:109] 

labels = np.argmax(y_test[100:109],axis=1)

#make predictions

predictions=model.predict(batch, verbose=1)

print(predictions)

for image in predictions: 
  
  print(np.sum(image))

class_result = np.argmax (predictions, axis=1) 

print (class_result)

fig, axs= plt.subplots (3, 3, figsize =(19,6)) 

fig.subplots_adjust(hspace = 1) 

axs=axs.flatten()

for i, img in enumerate(batch):

  for key, value in class_labels.items():
    
    if class_result[i] == key:
      
      title='Prediction: {}\nActual: {}'.format(class_labels[key], class_labels[labels[i]])
      
      axs[i].set_title(title)
      
      axs[1].axes.get_xaxis().set_visible(False) 
      
      axs[i].axes.get_yaxis().set_visible(False)
  
  axs[i].imshow(img)

plt.show()

![image](https://user-images.githubusercontent.com/78757768/135274419-fde3c732-06f9-4f6b-8f09-1a8c82103f70.png)


