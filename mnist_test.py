import os

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()



#plot the first image in the dataset
plt.imshow(X_train[0])
plt.show()

#check image shape
X_train[0].shape

print(X_train.shape)

#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
print(X_train.shape)

X_test = X_test.reshape(10000,28,28,1)

#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train[0]


#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy as a measure of model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train model
model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=3)

#show predictions for the first 3 images in the test set
model.predict(X_test[:4])

#show actual results for the first 3 images in the test set
y_test[:4]