# -------------------------------------------
#  CNNs classifier to classify on MNIST dataset
#  with comments for better understanding
#
#    (c) Keishi Ishihara
# -------------------------------------------

'''This code is also useful for better understanging how to use keras'''
from __future__ import print_function
import keras
from keras.datasets import mnist # keras module has mnist dataset
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from callbacks import LearningHistoryCallback
from get_best_model import getNewestModel

# configs
prefix = 'trial1' # for name of data # TODO: automatically dicide this name
batch_size = 256 # 128
num_classes = 10 # numbers are 10 types
epochs = 10 # epochs
debug = False # use small data
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
    x_train = x_train[:2000]
    y_train = y_train[:2000]

# normalize each image pixel values for all input data
x_train = x_train.astype('float32') / 255
x_val = x_val.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print('---')
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
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
    x = Conv2D(32, kernel_size=(3,3), activation='relu')(input_img) # conv layer1 with 3x3 kernel and relu function
    x = Conv2D(64, kernel_size=(3,3), activation='relu')(x) # conv layer2 with 3x3 kernel and relu func
    x = MaxPooling2D(pool_size=(2, 2))(x) # maxpooling layer, where pools features and convert to half size 
    x = Dropout(0.25)(x) # dropout layer
    x = Flatten()(x) # flatten the extracted features to input dense layer
    x = Dense(128, activation='relu')(x) # dense (fully conected) layer with 128 neurons, relu activation
    x = Dropout(0.5)(x) # dropout layer
    output = Dense(num_classes, activation='softmax')(x) # the output layer, 10 neurons, softmax activation

    # this creates a model
    cnn = Model(inputs=input_img, outputs=output)
    cnn.summary() # visualize model in console
    keras.utils.plot_model(cnn, to_file='models/model_cnn.png', show_shapes=True) # save model architecture as png

    # configure its learning process with compile() method
    cnn.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    # callbacks to be useful when training eg). monitoring training curves
    from keras.callbacks import ModelCheckpoint
    mc_cb = ModelCheckpoint( # this is for saving the model on each epochs when the model is better
                    filepath='models/model_{epoch:02d}_{val_loss:.2f}.hdf5',
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
cnn = getNewestModel('models')

# evaluate the model using test data
print('Evaluating CNN model..')
final_score = cnn.evaluate(x_test, y_test, verbose=1)

# evaluation & visualization here
import numpy as np 
y_test = np.argmax(y_test, axis=1) # reconvert from one-hot vector to label(scalar)
print('Predicting test set..')
# predict using test data, output is an array
prediction = cnn.predict(x_test, batch_size=256, verbose=1, steps=None)  
classified = np.argmax(prediction, axis=1) 
score = np.max(prediction, axis=1) * 100

#
misclassified_label_index = np.where((classified == y_test) == 0)[0]
arr1 = np.array([0 for i in range(10)]) #
arr2 = np.array([-1 for i in range(10)]) #

#
for i, xx in enumerate(misclassified_label_index):
    arr1[y_test[xx]] += 1
    if arr2[y_test[xx]] == -1:
        arr2[y_test[xx]] = xx

#
misclassify_plobability = arr1/np.sum(arr1) * 100
print('Misclassify probabilities >>')
for i, mp in enumerate(misclassify_plobability):
    print('class: {} -> {:.2f}'.format(i, mp))
print('---')
print('Test loss: {:.2f}'.format(final_score[0]))
print('Test accuracy: {:.2f}'.format(final_score[1]))

#
import matplotlib.pyplot as plt
f, axarr = plt.subplots(5, 2, figsize=(7,14))
for i in range(10):
    axarr[int(i/2), i%2].axis('off')
    axarr[int(i/2), i%2].set_title("Classified to {}, Score={:.1f}[%]\nMisclassify probability={:.2f}[%]".format(classified[arr2[i]], score[arr2[i]], misclassify_plobability[i]))
    axarr[int(i/2), i%2].imshow(x_test[arr2[i]].reshape(28,28), cmap='gray')

plt.tight_layout()
plt.savefig('results/{}_miss-classification.png'.format(prefix))
print('(result image is saved.)')