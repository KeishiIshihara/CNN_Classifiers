from __future__ import print_function
# --------------------------------------------
#  CNNs classifier to classify dogs and cats.
#  Train the model by transfer learning 
#  using VGG19 with imagenet weights
# --------------------------------------------

import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.applications.vgg19 import VGG19

import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dnn_modules.callbacks import LearningHistoryCallback, plot_confusion_matrix
from dnn_modules.get_best_model import getNewestModel


# ------------------------------------
# Loading data and Undestanding data

# PATH = '/Users/ishiharakeishi/Downloads/dogs-vs-cats/' # on mac
PATH = '/home/keishish/ishihara/uef/AI/dogs-vs-cats'
train_dir = os.path.join(PATH, 'train')
val_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
val_cats_dir = os.path.join(val_dir, 'cats')      # directory with our validation cat pictures
val_dogs_dir = os.path.join(val_dir, 'dogs')      # directory with our validation dog pictures
test_cats_dir = os.path.join(test_dir, 'cats')    # directory with our test cat pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')    # directory with our test dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
num_cats_val = len(os.listdir(val_cats_dir))
num_dogs_val = len(os.listdir(val_dogs_dir))
num_cats_test = len(os.listdir(test_cats_dir))
num_dogs_test = len(os.listdir(test_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val
total_test = num_cats_test + num_dogs_test

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)
print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print('total test cat images:', num_cats_test)
print('total test dog images:', num_dogs_test)

print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)
print("Total test images:", total_test)


# ---------------------------------
#  Prepare data generator

show_samples = False
prefix = 'trial1'
batch_size = 64
epochs = 30
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255)       # Generator for training data
val_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for validation data
test_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for test data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH), # all images are resized fixed shape
                                                           class_mode='binary')

val_data_gen = val_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=val_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH), 
                                                              class_mode='binary')

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH), 
                                                              class_mode='binary')


# --------------------------------------------------
#  Data sample plotting

def plotImages(images_arr, fname):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(fname)

if show_samples:
    sample_training_images, label = next(train_data_gen) # training sample
    plotImages(sample_training_images[:5], 'ex_train.png')
    print(label[:5])

    sample_val_images, label = next(val_data_gen) # validation sample
    plotImages(sample_val_images[:5], 'ex_val.png')
    print(label[:5])


# ------------------------------------------------
#  Define callbacks

mc_cb = ModelCheckpoint( # this is for saving the model on each epochs when the model is better
                filepath='models/model_{epoch:02d}_{val_loss:.2f}_'+prefix+'.hdf5',
                monitor='val_loss',
                verbose=1,
                save_best_only=True, 
                save_weights_only=False, # if True, save without optimazers to be used eg. retrain 
                mode='auto')
# for monitoring the training curves
lh_cb = LearningHistoryCallback(prefix=prefix)
callbacks = [mc_cb, lh_cb]


# ------------------------------------------------
#  Build Model from VGG19 trained with imagenet 

## base model: VGG19 with imagenet weights
input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)) # input tensor
base_model = VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor) # load vgg16 model

## Classifier (FC layers)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# freeze base model parameters
for layer in base_model.layers[:15]:
   layer.trainable = False

# configure the model
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9), 
              metrics=['accuracy'])

model.summary() # print model sammary in console
keras.utils.plot_model(model, to_file='fig_model/'+prefix+'_vgg19_tl.png', show_shapes=True) # save model architecture as png

# train the model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
    callbacks=callbacks,
    verbose=1
)

model, best_model_name = getNewestModel('models')


# evaluate the model using test data
print('Evaluating CNN model..')
final_train_score = model.evaluate_generator(train_data_gen, verbose=1)
final_test_score = model.evaluate_generator(test_data_gen, verbose=1)
print('---')
print('Train loss: {:.5f}'.format(final_train_score[0]))
print('Train accuracy: {:.5f}'.format(final_train_score[1]))
print('Test loss: {:.5f}'.format(final_test_score[0]))
print('Test accuracy: {:.5f}'.format(final_test_score[1]))


# make csv that specifies training details
import csv
header = ['prefix= ',prefix]
with open('results/{}_training_sammary.csv'.format(prefix),'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(header)
    writer.writerow([' '])
    writer.writerow(['[details]'])
    writer.writerow(['total epochs', 'batch size', 'IMG_HEIGHT', 'IMG_WIDTH'])
    writer.writerow([epochs, batch_size,IMG_HEIGHT, IMG_WIDTH])
    writer.writerow(['train sample #: ', total_train])
    writer.writerow(['val sample #: ', total_val])
    writer.writerow(['test sample #: ', total_test])
    writer.writerow([' '])
    writer.writerow(['Best model: ']+[best_model_name])
    writer.writerow(['Train loss: ']+[final_train_score[0]])
    writer.writerow(['Train accuracy: ']+[final_train_score[1]])
    writer.writerow(['Test loss: ']+[final_test_score[0]])
    writer.writerow(['Test accuracy: ']+[final_test_score[1]])