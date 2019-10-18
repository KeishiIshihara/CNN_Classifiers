# -----------------------------------------------
#  Best Trained Model Getter
#
#  Referenced from:
#   https://qiita.com/cvusk/items/7bcd3bc2e82bb45c9e9c
# -----------------------------------------------

from __future__ import print_function

from keras import models

import os, re, datetime, sys
from glob import glob


def getNewestModel(dirname, *, target='timestamp'):
    """Returns keras.models.model object and its filename.
    **Note: filename must contains epoch and loss eg).'model_e02_l0.02_trial4.hdf5'

    Usage:
    ```python
    model, fname = getNewestModel('models')

    ```
    # Arguments
        dirname: string name of directory where .hdf5 files are stored.
        target: (Optional) string name of target to select desired model. Defalt to 'timestamp'
         options: epoch
    
    # Return
        Model object, filename
    """

    targets = os.path.join(dirname, '*.hdf5') # targets = 'dirname/*.hdf5'
    files = [(f, os.path.getmtime(f)) for f in glob(targets)] # list of files
    # for f in files: print(f)

    if len(files) == 0:
        print('There is no model.')
        return None, None
        
    elif target == 'timestamp':
        sorted_lst = sorted(files, key=lambda files: files[1]) # sort by timestamp
        modelNewest = sorted_lst[-1]
        print('NewestModel (Best model) is {} (date: {})'.format(modelNewest[0], datetime.datetime.fromtimestamp(modelNewest[1])))
        model = models.load_model(modelNewest[0])
        return model, modelNewest[0]

    elif target == 'epoch':
        sorted_byepoch = sorted(files, key=lambda files: re.findall('_e(.*)_l', files[0])) # sort by epoch
        modelHighestEpoch = sorted_byepoch[-1]
        print('Model is {} (epoch: {})'.format(modelHighestEpoch[0], re.findall('_e(.*)_l', modelHighestEpoch[0])[0]))
        model = models.load_model(modelHighestEpoch[0])
        return model, modelHighestEpoch[0]
    
    elif target == 'loss':
        sorted_byloss = sorted(files, key=lambda files: re.findall('_l(.*)_', files[0])) # sort by loss
        modelLowestLoss = sorted_byloss[0]
        print('Model is {} (loss: {})'.format(modelLowestLoss[0], re.findall('_l(.*)_', modelLowestLoss[0])[0]))
        model = models.load_model(modelLowestLoss[0])
        return model, modelLowestLoss[0]
    
    else:
        print('Invalid target.')
        return None, None


if __name__ == '__main__':
    model, fname = getNewestModel('../mnist/models/', target='loss')
    print(fname)