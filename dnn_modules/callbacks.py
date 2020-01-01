# =================================================
#  Callback classes for visualizing training curves
#
#  (c) Keishi Ishihara
# =================================================

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

        os.makedirs('results/{}'.format(prefix), exist_ok=True)
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

            plt.savefig('results/{}/{}_training_curves.png'.format(self.prefix, self.prefix))
            plt.close()

    def training_log_csv(self, epoch):
        header = ['epoch','train_loss','val_loss','train_acc','val_acc']
        with open('results/{}/{}_training_log.csv'.format(self.prefix, self.prefix), 'w') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            writer.writerow(header)

            for i in range(0, epoch+1):
                writer.writerow([i+1, self.train_losses[i], self.val_losses[i], self.train_acc[i], self.val_acc[i]])


class ModelCheckpointSave(Callback):
    """Save model at the end of every epoch.
    Reference: https://keras.io/callbacks/#modelcheckpoint

    # Arguments
        filename: string, name of the model file to be saved.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        delete_old_model: 

    """
    def __init__(self, filename=None, prefix='', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, delete_old_model=False):
        self.filepath = filename
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.prefix = prefix
        self.delete_old_model = delete_old_model
        # super().__init__()

    def on_train_begin(self, logs=None):
        os.makedirs('models/{}'.format(self.prefix), exist_ok=True)
        self.monitor_op = np.less
        self.best = np.Inf
        self.previous_fpath = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # here self.filepath is expected to:
        # self.filepath= 'model_e{epoch:02d}_l{val_loss:.2f}_'+prefix+'.hdf5',
        filepath = 'models/' + self.prefix + '/' + self.filepath.format(epoch=epoch + 1, **logs)

        if self.save_best_only:
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                            ' saving model to %s'
                            % (epoch + 1, self.monitor, self.best,
                                current, filepath))
                self.best = current
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
                if self.delete_old_model and self.previous_fpath is not None:
                    os.remove(self.previous_fpath)

                self.previous_fpath = filepath

            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: %s did not improve from %0.5f' %
                            (epoch + 1, self.monitor, self.best))
        else:
            if self.verbose > 0:
                print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)

            if self.delete_old_model and self.previous_fpath is not None:
                os.remove(self.previous_fpath)

            self.previous_fpath = filepath
