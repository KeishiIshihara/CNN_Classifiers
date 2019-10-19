# ===============================================
#  Callback class for visualize training curves
#
#  (c) Keishi Ishihara
# ===============================================

from  keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import os, csv

class LearningHistoryCallback(Callback):
    """Vsualize training curves after every epoch.

    # Arguments
        prefix: string to be added to begining of file name.
        style: string, plt style to be adopted to figure.
            other options, see: print(plt.style.available)
        save_logs: if true, output training logs to csv file when every epoch ends.
        plot_steps: if true, plot training curves in detail.
    """
    
    def __init__(self, prefix='test', style='seaborn-darkgrid', save_logs=False, plot_steps=False):
        self.prefix = prefix
        self.save_logs = save_logs
        self.plot_in_detail = plot_steps

        os.makedirs('results', exist_ok=True)
        plt.style.use(style) # other option: seaborn, seaborn-colorblind

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.train_losses = []
        self.train_acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
        self.losses = [] # for every steps
        self.acc = [] # for every steps
    
    def on_batch_end(self, batch, logs={}):
        # only training loss and acc are avairable.
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # epoch: this valiable starts from 0.
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.train_losses.append(logs.get('loss'))
        self.train_acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))

        if self.save_logs:
            self.training_log_csv(epoch)

        # Before plotting ensure at least 2 epochs have passed
        if epoch >= 1:
            X = np.arange(0, epoch+1)

            if not self.plot_in_detail:
                _, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4)) # figsize=(10,4)
            else:
                _, (axL, axR, axExtra) = plt.subplots(ncols=3, figsize=(15,4)) # figsize=(10,4)

            # make pandas data frames
            data1 = pd.DataFrame({'train_loss': self.train_losses,
                                  'val_loss'  : self.val_losses    })
            data2 = pd.DataFrame({'train_acc' : self.train_acc,
                                  'val_acc'   : self.val_acc       })
            # plot
            sns.lineplot(data=data1, ax=axL)
            sns.lineplot(data=data2, ax=axR)
            # adjust losse curve
            axL.set_title('Training Loss')
            axL.set_xlabel('Epoch #')
            axL.set_ylabel('Loss')
            if epoch < 10:
                axL.set_xticks(X); axL.set_xticklabels(X+1)
            axL.legend(loc='upper right')
            # adjust accuracy curve
            axR.set_title('Training Accuracy')
            axR.set_xlabel('Epoch #')
            axR.set_ylabel('Accuracy')
            if epoch < 10:
                axR.set_xticks(X); axR.set_xticklabels(X+1)
            axR.legend(loc='lower right')

            # plot per step
            if self.plot_in_detail:
                X = np.arange(0, epoch+1)
                data3 = pd.DataFrame({'train_loss' : self.losses,
                                      'train_acc'  : self.acc   })
                line1 = sns.lineplot(data=data3.train_loss, color="g", label='train_loss', ax=axExtra)
                axExtra2 = axExtra.twinx()
                line2 = sns.lineplot(data=data3.train_acc, color="b", label='train_acc', ax=axExtra2)
                axExtra.set_title('Loss and Accuracy')
                axExtra.set_xlabel('Step #')
                axExtra.set_ylabel('Training Loss')
                axExtra2.set_ylabel('Training Accuracy')
                axExtra.legend(loc='best')
                axExtra2.legend(loc='best')

            plt.savefig('results/{}_training_curves.png'.format(self.prefix))
            plt.close()

    def training_log_csv(self, epoch):
        header = ['epoch','train_loss','val_loss','train_acc','val_acc']
        with open('results/{}_training_log.csv'.format(self.prefix), 'w') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            writer.writerow(header)

            for i in range(0, epoch+1):
                writer.writerow([i+1, self.train_losses[i], self.val_losses[i], self.train_acc[i], self.val_acc[i]])
