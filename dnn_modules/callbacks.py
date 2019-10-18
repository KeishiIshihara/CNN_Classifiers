# ===============================================
#  Callback class for visualize training curves
#
#  (c) Keishi Ishihara
# ===============================================

from  keras.callbacks import Callback
import numpy as np
import os
import matplotlib.pyplot as plt

class LearningHistoryCallback(Callback):
    """Vsualize training curves after every epoch.

    # Arguments
        prefix: string to be added to begining of file name.
        style: string, plt style to be adopted to figure.
    """
    
    def __init__(self, prefix='test', style='seaborn'):
        self.prefix = prefix
        self.style = style
        os.makedirs('results', exist_ok=True)

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.train_losses = []
        self.train_acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.train_losses.append(logs.get('loss'))
        self.train_acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))

        # Before plotting ensure at least 2 epochs have passed
        if epoch >= 1:
            X = np.arange(1, epoch+2)

            plt.style.use(self.style) # other option: seaborn, seaborn-colorblind
            _, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

            # plot losse curve
            axL.plot(X, self.train_losses, label='train_loss')
            axL.plot(X, self.val_losses, label='val_loss')
            axL.set_title('Training Loss')
            axL.set_xlabel('Epoch #')
            axL.set_ylabel('Loss')
            if epoch < 10:
                axL.set_xticks(X)
                axL.set_xticklabels(X)
            axL.legend(loc='upper right')

            #  plot accuracy curve
            axR.plot(X, self.train_acc, label='train_acc')
            axR.plot(X, self.val_acc, label='val_acc')
            axR.set_title('Training Accuracy')
            axR.set_xlabel('Epoch #')
            axR.set_ylabel('Accuracy')
            if epoch < 10:
                axR.set_xticks(X)
                axR.set_xticklabels(X)
            axR.legend(loc='lower right')
            plt.savefig('results/{}_training_curves.png'.format(self.prefix))
            plt.close()



# =======================================================
#  Plot confusion matrix using matplotlib
#  by modifying following referece:
#
#  Reference: Confusion matrix — scikit-learn 0.21.3 documentation
#  https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
# =======================================================

import matplotlib as mpl
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          prefix='test',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    mpl.rcParams.update(mpl.rcParamsDefault) # clear current plt style defined before
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    _, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(cm.shape[1]), minor=False)
    ax.set_yticks(np.arange(cm.shape[0]) , minor=False)
    ax.set_ylim(9.5,-0.5)
    ax.set(# xticks=np.arange(cm.shape[1]),
           # yticks=np.arange(cm.shape[0])+0.5,
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # fig.tight_layout()
    plt.savefig('results/{}_confusion_matrix.png'.format(prefix))
    plt.close