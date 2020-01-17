from __future__ import print_function
# ===========================================================
#  CNNs classifier to classify dogs and cats.
#  Train the model by transfer learning
#  using VGG19 pretrained on Imagenet.
#
#  Reference:
#  https://www.tensorflow.org/tutorials/images/classification
# ===========================================================

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.applications.vgg19 import VGG19

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dnn_modules.callbacks import LearningHistoryCallback, ModelCheckpointSave
from dnn_modules.get_best_model import getNewestModel


# arguments
parser = argparse.ArgumentParser(description='CNN trainer on Dogs vs. Cats dataset')
parser.add_argument('-s','--show-samples', action='store_true', default=False, help='Save figure of data sample')
parser.add_argument('-e','--epochs', default=10, type=int, help='Number of epochs you run (default 10)')
parser.add_argument('-b','--batch-size', default=64, type=int, help='Batch size (default 64)')
parser.add_argument('-p','--prefix', default='test', type=str, help='prefix to be added to result filenames (default \'test\')')
parser.add_argument('--plot-steps', action='store_true', default=False, help='plot in detail')
parser.add_argument('--save-logs', action='store_true', default=True, help='save training logs to csv file')
args = parser.parse_args()


# -------------------------------------------
#       Load data and Undestand data
# -------------------------------------------
PATH = os.path.dirname(os.path.abspath(__file__)) + '/dataset'
train_dir = os.path.join(PATH, 'train')
val_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
val_cats_dir = os.path.join(val_dir, 'cats')      # directory with our validation cat pictures
val_dogs_dir = os.path.join(val_dir, 'dogs')      # directory with our validation dog pictures
test_cats_dir = os.path.join(test_dir, 'cats')    # directory with our test cat pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')    # directory with our test dog pictures

# length of each dataset
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
num_cats_val = len(os.listdir(val_cats_dir))
num_dogs_val = len(os.listdir(val_dogs_dir))
num_cats_test = len(os.listdir(test_cats_dir))
num_dogs_test = len(os.listdir(test_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val
total_test = num_cats_test + num_dogs_test

print('training cat images #:    ', num_cats_tr)
print('training dog images #:    ', num_dogs_tr)
print('validation cat images #:  ', num_cats_val)
print('validation dog images #:  ', num_dogs_val)
print('test cat images #:        ', num_cats_test)
print('test dog images #:        ', num_dogs_test)

print('--')
print('Total training images #:  ', total_train)
print('Total validation images #:', total_val)
print('Total test images #:      ', total_test)
print('--')


# -------------------------------------
#         Define data generators
# -------------------------------------
# he entire dataset is too learge to load into RAM (probably),
# it is necessary to load them in batch data.
# Here, the generators are defined for train, validation and test data.

show_samples = args.show_samples # if true, produce 5 sample images as png.
prefix = args.prefix             # prefix
batch_size = args.batch_size     # bath size (default 64)
epochs = args.epochs             # epochs (default 10)
IMG_HEIGHT = 150                 # all images will be resized
IMG_WIDTH = 150                  # to this size to feed neural networks

# make folders for storing results
os.makedirs('fig_model', exist_ok=True)
os.makedirs('models/{}'.format(prefix), exist_ok=True)
os.makedirs('results/{}'.format(prefix), exist_ok=True)

train_image_generator = ImageDataGenerator(rescale=1./255)   # Generator for training data
val_image_generator = ImageDataGenerator(rescale=1./255)     # Generator for validation data
test_image_generator = ImageDataGenerator(rescale=1./255)    # Generator for test data

# these instances produce a batch data from directory
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, 
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = val_image_generator.flow_from_directory(batch_size=batch_size,
                                                       directory=val_dir,
                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       class_mode='binary')

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                         directory=test_dir,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode='binary')


# --------------------------------------
#         Sample data plotting
# --------------------------------------
def plotImages(images_arr, fname):
    _, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(fname)

if show_samples:
    # training sample
    sample_training_images, label = next(train_data_gen)
    plotImages(sample_training_images[:5], 'ex_train.png')
    print(label[:5])
    # validation sample
    sample_val_images, label = next(val_data_gen)
    plotImages(sample_val_images[:5], 'ex_val.png')
    print(label[:5])


# -----------------------------------------
#            Define callbacks
# -----------------------------------------
# this is for saving the model on each epoch ends when only the model is improved
# mc_cb = ModelCheckpoint(filepath='models/'+prefix+'/model_e{epoch:02d}_l{val_loss:.2f}_'+prefix+'.hdf5',
#                         monitor='val_loss',
#                         verbose=1,
#                         save_best_only=True,
#                         save_weights_only=False, # if True, save without optimazers to be used eg. retrain
#                         mode='auto')

# Not yet tested
mc_cb = ModelCheckpointSave(filename='model_e{epoch:02d}_l{val_loss:.2f}_'+prefix+'.hdf5',
                            prefix=prefix,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            delete_old_model=True)

# for monitoring the training curves
lh_cb = LearningHistoryCallback(prefix=prefix, style='ggplot', save_logs=args.save_logs, plot_steps=args.plot_steps)
callbacks = [mc_cb, lh_cb]


# -------------------------------------------------------
#    Build model based on VGG19 pretrained on ImageNet
# -------------------------------------------------------
## base model: VGG19 with weights
input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)) # input tensor
base_model = VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor) # load VGG19 model

# classifier (Output layers)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x) # Outputs are between 0 to 1. 0 means cat, 1 means dog.
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


# -------------------------------------
#           Train the model
# -------------------------------------
# train the model
model.fit_generator(train_data_gen, # data generator object
                    steps_per_epoch=total_train // batch_size, # number of times to update weight per epoch. 
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size, # // means truncate division
                    callbacks=callbacks,
                    verbose=1)


# ------------------------------------
#          Evaluate the model
# ------------------------------------
# load the best model from directory named models to evalate the performance
model, best_model_name = getNewestModel('models/{}'.format(prefix))

# evaluate the model using test data
print('Evaluating CNN model..')
final_train_score = model.evaluate_generator(train_data_gen, verbose=1)
final_test_score = model.evaluate_generator(test_data_gen, verbose=1)
print('---')
print('Train loss: {:.5f}'.format(final_train_score[0]))
print('Train accuracy: {:.5f}'.format(final_train_score[1]))
print('Test loss: {:.5f}'.format(final_test_score[0]))
print('Test accuracy: {:.5f}'.format(final_test_score[1]))

# sammarize training details and results into csv
import csv
header = ['prefix:',prefix]
with open('results/{}/{}_training_summary.csv'.format(prefix,prefix),'w') as f:
    writer = csv.writer(f, delimiter='\t', lineterminator='\n')
    writer.writerow(header)
    writer.writerow([' '])
    writer.writerow(['[details]'])
    writer.writerow(['total epochs', 'batch size', 'IMG_HEIGHT', 'IMG_WIDTH'])
    writer.writerow([epochs, batch_size, IMG_HEIGHT, IMG_WIDTH])
    writer.writerow(['train sample #:', total_train])
    writer.writerow(['val sample #:', total_val])
    writer.writerow(['test sample #:', total_test])
    writer.writerow([' '])
    writer.writerow(['Best model:']+[best_model_name])
    writer.writerow(['Train loss:']+[final_train_score[0]])
    writer.writerow(['Train accuracy:']+[final_train_score[1]])
    writer.writerow(['Test loss:']+[final_test_score[0]])
    writer.writerow(['Test accuracy:']+[final_test_score[1]])