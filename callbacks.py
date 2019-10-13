# ------------------------------------------
#  Callback classes for training with Keras
#
#  (c) Keishi Ishihara
# ------------------------------------------
import os
import numpy as np
from  keras.callbacks import Callback
import matplotlib.pyplot as plt


class LearningHistoryCallback(Callback):
    """Callback class for visualizing training curves"""
    def __init__(self, prefix='test'):
        self.prefix = prefix
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

            N = np.arange(0, len(self.train_losses))

            plt.style.use("ggplot") # other option: seaborn, seaborn-colorblind
            fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

            axL.plot(N, self.train_losses, label='train_loss')
            axL.plot(N, self.val_losses, label='val_loss')
            axL.set_title('Training Loss')
            axL.set_xlabel('Epoch #')
            axL.set_ylabel('Loss')
            axL.set_xticks(N)
            axL.set_xticklabels(N)
            axL.legend(loc='upper right')

            axR.plot(N, self.train_acc, label='train_acc')
            axR.plot(N, self.val_acc, label='val_acc')
            axR.set_title('Training Accuracy')
            axR.set_xlabel('Epoch #')
            axR.set_ylabel('Accuracy')
            axR.set_xticks(N)
            axR.set_xticklabels(N)
            axR.legend(loc='lower right')
            plt.savefig('results/{}_result.png'.format(self.prefix))
            plt.close()

