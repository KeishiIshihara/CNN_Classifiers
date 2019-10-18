# Classifiers with CNNs in Keras
For better understanding Convolutional Neural Networks for classification tasks by implementing appropreate model using Keras.
Train on MNIST dataset from scrach, and on dogs vs. cats dataset by adopting VGG19 model with weights pretrained on ImageNet dataset.  


### Environments
- macOS High Sierra 10.13.6 or Ubuntu 18.04 with Geforce GTX 1080
- Python 3.6.7
- Keras 2.3.0
- tensorflow(-gpu) 1.14.0 (backend)

In your pyenv environment, run 
```bash
$ pip install -r requirements.txt
```

### Prerequisites
Before training `dogs vs. cats`, you suppose to have the `dogs vs. cats` dataset (Otherwise download it from Kaggle).  
You can get ready to train by running:
```bash
# cd dogs-vs-cats/
$ python prepare_dataset.py --souce-dir #path to your folder where dataset exists.
```
**Note: this creates a folder for storing dogs vs. cats dataset as subdirectory of this repository.

## On MNIST dataset
Train CNN model by running,
```bash
# cd mnist/
$ python mnist_keras.py -e 10 -b 32 --prefix trial1
```
It gets around 99.47% test accuracy after 14 epoch.

### Training curves
![training curves](mnist/results/trial5_training_curves.png)
*Note: [machine learning - Higher validation accuracy, than training accurracy using Tensorflow and Keras - Stack Overflow](https://stackoverflow.com/questions/43979449/higher-validation-accuracy-than-training-accurracy-using-tensorflow-and-keras)

### Normarized confusion matrix

<img src="mnist/results/trial5_confusion_matrix.png" width="500px">

### Misclassified examples (picked the most often occurs one)

<img src="mnist/results/trial5_misclassification.png" width="700px">


## On Dog vs. Cats dataset
Use VGG19 with weights pretrained ImageNet dataset as base model to extract features.  
Train classifier by running,
```python
# cd dogs-vs-cats/
$ python train_keras.py -e 10 -b 64 --prefix trial1
```
**Note: I don't recommend you to set batch size too large. My pc suddenly shut down because PSU resource was not enough.  

It gets around 96.20% test accuracy after 27 epoch.

### Training curves
![training curves](dogs-vs-cats/results/trial2_training_curves.png)



## References
- MNIST:
    - [Confusion matrix — scikit-learn 0.21.3 documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)
    - [keras/mnist_cnn.py at master · keras-team/keras](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
    - [Kerasのモデルは学習完了時のものが最良とは限らない - Qiita](https://qiita.com/cvusk/items/7bcd3bc2e82bb45c9e9c)
- Dogs vs. Cats:
    - [Dogs vs. Cats | Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)
    - [Applications - Keras Documentation](https://keras.io/applications/#vgg19)
    - [Image classification  |  TensorFlow Core](https://www.tensorflow.org/tutorials/images/classification)
