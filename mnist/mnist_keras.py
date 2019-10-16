from __future__ import print_function

# ----------------------------------------------
#  CNNs classifier to classify on MNIST dataset
#  with comments for better understanding
#
#    (c) Keishi Ishihara
# ----------------------------------------------

'''This might be helpful also for coding with keras'''
'''After 30 epoch it gets 0.99470 accuracy (loss=0.01696) with model model is trail4_30e_model_20_0.02.hdf5'''
'''(4 seconds per epoch with GTX 1080)'''

import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

import keras
from keras.datasets import mnist # keras module has mnist dataset
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

import os, sys 
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dnn_modules.callbacks import LearningHistoryCallback, plot_confusion_matrix
from dnn_modules.get_best_model import getNewestModel

# configs
prefix = 'trail5' # for name of data # TODO: automatically dicide this name
batch_size = 128 # 128
num_classes = 10 # numbers are 10 types
epochs = 15 # epochs
debug = False # use small data for debugging
only_evaluate = False # only evaluate the already trained model without train new model
img_rows, img_cols = 28, 28 # input image dimensions

# load mnist dataset splited between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# the original data format is (60000, 28, 28).
# chanels' dimention is need to input the network,
# so here, it converts to (60000, 28, 28, 1).
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# split data to make validation data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

if debug:
    x_train = x_train[:5000]
    y_train = y_train[:5000]

# normalize each image pixel values for all input data
x_train = x_train.astype('float32') / 255
x_val = x_val.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print('---')
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'val samples')
print(x_test.shape[0], 'test samples')

# one-hot encord each label data
# this means that convert each numbers to one-hot vectors
# eg). '3' -> [0,0,0,1,0,0,0,0,0,0]
y_train = keras.utils.to_categorical(y_train, num_classes) # 48000 samples
y_val = keras.utils.to_categorical(y_val, num_classes)     # 12000 samples
y_test = keras.utils.to_categorical(y_test, num_classes)   # 10000 samples


if not only_evaluate:
    # define CNN Architecture using Functional API
    input_img = Input(shape=input_shape) # this returns a tensor
    x = Conv2D(32, kernel_size=(3,3), activation='relu')(input_img) # conv layer with 3x3 kernel and relu function
    x = Conv2D(64, kernel_size=(3,3), activation='relu')(x) # conv layer with 3x3 kernel and relu func
    x = MaxPooling2D(pool_size=(2, 2))(x) # maxpooling layer, where pools features and convert to half size 
    x = Dropout(0.25)(x) # dropout layer
    x = Conv2D(64, kernel_size=(3,3), activation='relu')(x) # conv layer with 3x3 kernel and relu func
    x = Conv2D(64, kernel_size=(3,3), activation='relu')(x) # conv layer with 3x3 kernel and relu func
    x = MaxPooling2D(pool_size=(2, 2))(x) # maxpooling layer, where pools features and convert to half size 
    x = Dropout(0.25)(x) # dropout layer
    x = Flatten()(x) # flatten the extracted features to input dense layer
    x = Dense(128, activation='relu')(x) # dense (fully conected) layer with 128 neurons, relu activation
    x = Dropout(0.5)(x) # dropout layer
    output = Dense(num_classes, activation='softmax')(x) # the output layer, 10 neurons, softmax activation

    # this creates a model
    cnn = Model(inputs=input_img, outputs=output)
    cnn.summary() # visualize model in console
    keras.utils.plot_model(cnn, to_file='fig_model/'+prefix+'_model_cnn.png', show_shapes=True) # save model architecture as png

    # configure its learning process with compile() method
    cnn.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    # callbacks to be useful when training eg). monitoring training curves
    mc_cb = ModelCheckpoint( # this is for saving the model on each epochs when the model is better
                    filepath='models/model_{epoch:02d}_{val_loss:.2f}_'+prefix+'.hdf5',
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True, 
                    save_weights_only=False, # if True, save without optimazers to be used eg. retrain 
                    mode='auto')
    # for monitoring the training curves
    lh_cb = LearningHistoryCallback(prefix=prefix)
    # for chainging training rates, define callback here
    # if you want to use tensorboard, define it here
    callbacks = [mc_cb, lh_cb]

    # train the model
    history = cnn.fit(x_train, y_train, # training data and label
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val, y_val), # validation data
                    callbacks=callbacks
                    )

    del cnn # delete the model to load the best model (this is not necessary)


# load the best model from directory named models
cnn, best_model_name = getNewestModel('models')

# evaluate the model using test data
print('Evaluating CNN model..')
final_train_score = cnn.evaluate(x_train, y_train, verbose=1)
final_test_score = cnn.evaluate(x_test, y_test, verbose=1)

# evaluation & visualization here
y_test = np.argmax(y_test, axis=1) # reconvert from one-hot vector to label(scalar)
print('Predicting test set..')
# predict using test data, output is an array which has all predicted data
prediction = cnn.predict(x_test, batch_size=256, verbose=1, steps=None)
classified = np.argmax(prediction, axis=1)
score = np.max(prediction, axis=1) * 100

# confution matrix
print('Confusion Matrix')
cm = confusion_matrix(y_test, classified) # sklearn method to calculate CM
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize
# print(cm)

# classification report
print('Classification Report')
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(classification_report(y_test, classified, target_names=target_names))

# plot normalized confusion matrix
plot_confusion_matrix(y_test, classified, classes=np.array(target_names),
                      prefix=prefix,
                      normalize=True)

# pick the most common misclassified data in each classes
misclassified_class = np.array([0 for i in range(num_classes)])
misclassified_probability = np.array([0. for i in range(num_classes)])
for i, m in enumerate(cm):
    misclassified_class[i] = m.argsort()[::-1][1] # secondly largest probability of misclassify class
    misclassified_probability[i] = m[misclassified_class[i]] * 100 # secondly largest probability of misclassify

# sample one misclassified data of each classes
misclassified_label_index = np.where((classified == y_test) == 0)[0]
sample_idx = []
for i in range(num_classes):
    for xx in misclassified_label_index:
        if y_test[xx] == i and classified[xx] == misclassified_class[i]:
            sample_idx.append(xx)
            break
print('sampled indices: ',sample_idx)


# plot result
f, axarr = plt.subplots(5, 2, figsize=(7,14))
for i in range(len(cm)):
    axarr[int(i/2), i%2].axis('off')
    axarr[int(i/2), i%2].set_title("Classified to {}(={:.1f}%)\n Its probability={:.2f}%".format(classified[sample_idx[i]], score[sample_idx[i]], misclassified_probability[i]))
    axarr[int(i/2), i%2].imshow(x_test[sample_idx[i]].reshape(img_rows,img_cols), cmap='gray')
plt.tight_layout()
plt.savefig('results/{}_miss-classification.png'.format(prefix))
print('(result image is saved.)')

print('---')
print('Ttain loss: {:.5f}'.format(final_train_score[0]))
print('Train accuracy: {:.5f}'.format(final_train_score[1]))
print('Test loss: {:.5f}'.format(final_test_score[0]))
print('Test accuracy: {:.5f}'.format(final_test_score[1]))


# make csv that specifies training details
import csv
header = ['prefix: ',prefix]
with open('results/{}_training_detail.csv'.format(prefix),'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(header)
    writer.writerow([''])
    writer.writerow(['[details]'])
    writer.writerow(['epochs', 'batch size', 'IMG_HEIGHT', 'IMG_WIDTH'])
    writer.writerow([epochs, batch_size, img_rows, img_cols])
    writer.writerow(['train sample #: ', x_train.shape[0]])
    writer.writerow(['val sample #: ', x_val.shape[0]])
    writer.writerow(['test sample #: ', x_test.shape[0]])

    writer.writerow([''])
    writer.writerow(['Best model: ']+[best_model_name])
    writer.writerow(['Train loss: ']+[final_train_score[0]])
    writer.writerow(['Train accuracy: ']+[final_train_score[1]])
    writer.writerow(['Test loss: ']+[final_test_score[0]])
    writer.writerow(['Test accuracy: ']+[final_test_score[1]])

