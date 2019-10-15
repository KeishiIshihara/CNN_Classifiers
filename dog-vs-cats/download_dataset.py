# --------------------------------------
#  Download dataset
#
#
#
# --------------------------------------

import urllib.request
import zipfile
import os

source_dir = "./tmp/trainData"
train_dir = "./dataset/trainData"
valid_dir = "./dataset/validationData"


os.makedirs("%s/dogs" % train_dir, exist_ok=True)
os.makedirs("%s/cats" % train_dir, exist_ok=True)
os.makedirs("%s/dogs" % valid_dir, exist_ok=True)
os.makedirs("%s/cats" % valid_dir, exist_ok=True)

# Kaggleよりデータをダウンロードする
url  = "https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/download/train.zip" 
urllib.request.urlretrieve(url, './tmp/trainData.zip')

# データの解凍
with zipfile.ZipFile('./tmp/trainData.zip', 'r') as f:
    f.extractall('./train/')

# 訓練用データの格納
for i in range(1000):
    os.rename("%s/dog.%d.jpg" % (source_dir, i + 1), "%s/dogs/dog%04d.jpg" % (train_dir, i + 1))
    os.rename("%s/cat.%d.jpg" % (source_dir, i + 1), "%s/cats/cat%04d.jpg" % (train_dir, i + 1))

# 検証用データの格納
for i in range(400):
    os.rename("%s/dog.%d.jpg" % (source_dir, 1000 + i + 1), "%s/dogs/dog%04d.jpg" % (valid_dir, i + 1))
    os.rename("%s/cat.%d.jpg" % (source_dir, 1000 + i + 1), "%s/cats/cat%04d.jpg" % (valid_dir, i + 1))