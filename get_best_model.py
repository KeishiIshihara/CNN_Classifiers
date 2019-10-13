from __future__ import print_function

import os
from glob import glob
from keras import models

def getNewestModel(dirname):
    target = os.path.join(dirname, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    if len(files) == 0:
        return False
    else:
        newestModel = sorted(files, key=lambda files: files[1])[-1]
        print('newestModel (best model) is',newestModel[0])
        model = models.load_model(newestModel[0])
        return model