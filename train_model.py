import csv
import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy import misc
import tensorflow as tf
import keras
from keras.utils import multi_gpu_model
from skimage.util import random_noise
from skimage import io, color, exposure, filters, img_as_ubyte
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from random import shuffle
from keras.utils import plot_model

def read_logs(dir):
    '''
    Read
    :param dir:
    :return:
    '''
    lines = []
    csv_file = dir + "/" + 'result_data.csv'
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines



def newGenerator( rows, batch_size=64 ):
    num_samples = len(rows)
    while 1:  # Loop forever so the generator never terminates
        shuffle(rows)
        for offset in range(0, num_samples, batch_size):
            images = []
            angles = []
            batch_rows = rows[offset:offset + batch_size]
            for row in batch_rows:
                images.append(cv2.imread(row[0]))
                angles.append(float(row[1]))

            X = np.array(images)
            y = np.array(angles)

            yield sklearn.utils.shuffle(X, y)

def simpleNetwork():
    model = Sequential()
    model.add(Lambda(lambda x: x/ 255.0 - 0.5, input_shape=(66,200,3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model


def nvidiaNetwork():
    model = Sequential()
    model.add(Lambda(lambda x: x/ 255.0 - 0.5, input_shape=(66,200,3)))
    model.add(Convolution2D(24,kernel_size=(5,5),strides=[2,2],activation='relu'))
    model.add(Convolution2D(36,kernel_size=(5,5),strides=[2,2],activation='relu'))
    model.add(Convolution2D(48,kernel_size=(5,5),strides=[2,2],activation='relu'))
    model.add(Convolution2D(64,kernel_size=(3,3),activation='relu'))
    model.add(Convolution2D(64,kernel_size=(3,3),activation='relu'))
    model.add(Flatten())
    #model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


print("Read drive data...")
samples = read_logs("data")

print("Split data...")
train_data, validation_data = train_test_split(samples, test_size=0.1)

print(len(train_data),len(validation_data))


train_generator = newGenerator(train_data, batch_size=256)
validation_generator = newGenerator(validation_data, batch_size=256)


#model = simpleNetwork()
model = nvidiaNetwork()
model.summary()
plot_model(model, to_file='out_images/model.png')


parallel_model = multi_gpu_model(model, gpus=5)

parallel_model.compile(optimizer=Adam(lr=1e-3),loss='mse',metrics=['accuracy','mae'])

history_object = parallel_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=len(train_data),
                    validation_steps=len(validation_data),
                    epochs=5,
                    verbose=1
                    )

parallel_model.save("model.h5")

print(history_object.history.keys())


plt.figure(figsize=(20,20))
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("out_images/loss_history.jpg")
