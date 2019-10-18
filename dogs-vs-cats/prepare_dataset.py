from __future__ import print_function
# =========================================
#   Prepare Dogs vs. Cats dataset 
#
#    (c) Keishi Ishihara
# =========================================

import zipfile
import argparse
import os, sys

# arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('-s','--source-dir', default=None, type=str, help='Path to your folder where zip of dogs vs. cats dataset is.')
args = parser.parse_args()


# -------------------------
#        Preprocess
# -------------------------
# change current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# configs
source_dir = args.source_dir # eg. download
base_dir   = os.path.join(os.getcwd(), 'dataset')
# base_dir     = '/Users/ishiharakeishi/Downloads/dataset/'
train_dir  = 'train'
valid_dir  = 'valid'
test_dir   = 'test'
dirs = [train_dir, valid_dir, test_dir]

_train = 0.8
_val   = 0.1
_test  = 0.1
rates = [_train, _val, _test]

print('[Infomations]')
print('  source dir:  ',source_dir)
print('  base dir:    ',base_dir)
print('  train:val:test = {}:{}:{}'.format(_train,_val,_test))

if os.path.exists(os.path.join(source_dir, 'dogs-vs-cats.zip')) : print('  (Zip file found.)')
else: 
    print('[Error] Zip file not found.')
    sys.exit()

print('\n** This will create dataset directory under {}'.format(os.getcwd()))
x = input('Enter yes(y)>>')
if not (x == 'yes' or x == 'y'):
    print('please enter \'yes\'')
    sys.exit()
os.makedirs(base_dir, exist_ok=True)
print('---')


# --------------------------
#       Unzip dataset
# --------------------------
print('step1: unziping dogs-vs-cats.zip..')
try:
    with zipfile.ZipFile(os.path.join(source_dir,'dogs-vs-cats.zip'), 'r') as f:
        f.extractall(base_dir)

    print('step2: unziping train.zip..')
    with zipfile.ZipFile(os.path.join(base_dir,'train.zip'), 'r') as f:
        f.extractall(base_dir)

except TypeError as err:
    print('[Error] please specify your valid path to dataset in command-line argument')
    sys.exit()

os.rename(os.path.join(base_dir,'train'), os.path.join(base_dir,'train_original'))


# ---------------------------
#       Create folders
# ---------------------------
print('step3: creating folders..')
for i in range(len(dirs)):
    os.makedirs(os.path.join(base_dir, dirs[i], 'cats'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, dirs[i], 'dogs'), exist_ok=True)

files = os.listdir(os.path.join(base_dir,'train_original'))  
num_files = int(len(files)/2)
print('    (total file # {})'.format(num_files))


# -----------------------------------
#    Move files to certain folder
# -----------------------------------
print('step4: preparing each data..')
start, end = 0, 0
for i in range(len(dirs)):
    end += int(num_files * rates[i])
    print('  - (start={}, end={})'.format(start, end))
    for j in range(start, end):
        os.rename('{}/dog.{}.jpg'.format(os.path.join(base_dir,'train_original'), j), 
                  '{}/dogs/dog{}.jpg'.format(os.path.join(base_dir, dirs[i]), j))
        os.rename('{}/cat.{}.jpg'.format(os.path.join(base_dir,'train_original'), j), 
                  '{}/cats/cat{}.jpg'.format(os.path.join(base_dir, dirs[i]), j))
    print('  - # {}: {}'.format(dirs[i],end-start))
    start = end


os.rmdir(os.path.join(base_dir,'train_original')) # remove empty folder
print('Done.')