# This file builds, trains, and tests a replication of LeNet5 on MNIST data set,
# but with the following improvements: ReLU activation, softmax activation, Adam optimizer

import keras
import keras.optimizers
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import copy

from sklearn.model_selection import train_test_split
import keras
from keras.layers import Activation
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten

# Data preparation

df_train = pd.read_csv('MNIST_data/mnist_train.csv')
df_test = pd.read_csv('MNIST_data/mnist_test.csv')

labels_train = df_train.iloc[1:, 0].values
pixels_train = df_train.iloc[1:, 1:].values
labels_test = df_test.iloc[1:, 0].values
pixels_test = df_test.iloc[1:, 1:].values

# combine testing & training data
labels = np.concatenate((labels_train, labels_test))
pixels = np.concatenate((pixels_train, pixels_test))

num_images = len(labels)
pixels = pixels.reshape(num_images, 28, 28)

# resize to 32x32, normalize values to be between 0-1
pixels_resized = np.zeros((num_images, 32, 32))  
pixels_resized[:, 2:30, 2:30] = pixels
pixels = pixels_resized / 255.0

background_val = -0.1 # Value taken from LeNet5 paper
foreground_val = 1.175 # Value taken from LeNet5 paper
# These values were chosen to make the mean input ~0 and 
# the mean variance ~1, which accelerates learning

threshold = 0.25
pixels[pixels < threshold] = background_val # set all "white" pixels to background value
pixels[pixels >= threshold] = foreground_val  # set all "black" pixels to foreground value

# split data
X_train, X_test, y_train, y_test = train_test_split(pixels, labels, test_size = 0.14, random_state=42)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


# create model
def generate_model(): 

    model = keras.models.Sequential()

    model.add(Conv2D(6, kernel_size=(5,5), activation='relu', padding='same')) # C1
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2))) # S2
    model.add(Conv2D(16, kernel_size=(5,5), activation='relu', padding='same')) # C3
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2))) # S4
    model.add(Conv2D(120, kernel_size=(5,5), activation='relu', padding='same')) # C5

    model.add(Flatten())
    model.add(Dense(84, activation='relu')) # F6
    model.add(Dense(10, activation='softmax')) # OUTPUT

    return model

model = generate_model()

# Training
adam = keras.optimizers.Adam(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

print("Beginnign training...\n")
model.fit(X_train, y_train, batch_size=32, epochs=20)

model.save('LeNet5_improved_replica.keras')
print("Training complete. Beginning testing...\n")


# Testing
loss, accuracy = model.evaluate(X_test, y_test)
print("Testing Complete. \nTesting accuracy = ", accuracy)