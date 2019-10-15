


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg16 import VGG16
import os, sys
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------
# Loading data and Undestanding data

PATH = '/Users/ishiharakeishi/Downloads/dogs-vs-cats/'
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation') # validation is from google's dataset
test_dir = os.path.join(PATH, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')            # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')            # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures


num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)


# ---------------------------------
#  Prepare data generator

batch_size = 128
epochs = 3
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255)       # Generator for training data
validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH), # all images are resized fixed shape
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH), 
                                                              class_mode='binary')


# This function will plot images in the form of a grid 
# with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# sample_training_images, hoge = next(train_data_gen)
# plotImages(sample_training_images[:5])
# print(label[:5])

# sys.exit()

# ------------------------------------------------
#  Build Model from VGG16 trained with imagenet 


# VGG16(model & weight)をインポート
input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
print(input_tensor)
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
# base_model = VGG16(include_top=False, weights='imagenet')
# base_model.summary()

""" ## Using BOTTLENECK FEATURES
# generate bottleneck feature data using VGG16
bottleneck_feature_train = model.predict_generator(train_data_gen, verbose=1)    # training data
bottleneck_feature_validation = model.predict_generator(val_data_gen, verbose=1) # validation data 

# save bottleneck features
train_file_name = 'train'
validation_file_name = 'validation'
np.save(PATH + 'bottleneck_features/' + train_file_name, bottleneck_feature_train)            # traning data
np.save(PATH + 'bottleneck_features/' + validation_file_name, bottleneck_feature_validation)  # validation data

# load bottleneck features
train_data  = np.load(PATH + 'bottleneck_features/'  + train_file_name)
len_input_samples = len(train_data)
train_labels = np.array([0] * int(len_input_samples/2) + [1] * int(len_input_samples / 2))

validation_data = np.load(PATH + 'bottleneck_features/'  + validation_file_name)
validation_labels = np.array([0] * int(n_validation_samples / 2 *32) + [1] * int(n_validation_samples / 2 * 32))
"""

# # Classifier (FC layers)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# predictions = Dense(N_CATEGORIES, activation='softmax')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers[:15]: # freeze base model
   layer.trainable = False
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
model.compile(loss='binary_crossentropy', 
              optimizer=SGD(lr=1e-4, momentum=0.9), 
              metrics=['accuracy'])

model.summary()
keras.utils.plot_model(model, to_file='models/'+prefix+'_model_cnn.png', show_shapes=True) # save model architecture as png

# sys.exit()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()