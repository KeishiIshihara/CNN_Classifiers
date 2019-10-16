# CNN Classifier with Keras

## TODO
- [x] automatically output the training conditions to csv file eg. epochs, data amounts, 
- [ ] automatically set the prefix
- [ ] make callback for changing learning rate
- [x] comment each step
- [x] make a generator for other dataset which has more amount
- [x] use ImageGenerator to process image data before training
- [x] visualize confusion matrix

### Environments
- macOS High Sierra 10.13.6
- Python 3.6.7
- Keras 2.3.0
- tensorflow(-gpu) 1.14.0 (backend)

In your pyenv environment, run 
```
$ pip install -r requirements.txt
```

## On MNIST dataset
Train CNN model with running,
```
$ python mnist_keras.py
```
It gets around 99.47% test accuracy after 14 epoch.

- Training curves
[training curves](mnist/results/trial5_training_curves.png)
- Normarized confusion matrix
[cunfution matrix](mnist/results/trial5_confusion_matrix.png)
- Missclassified examples (picked the most often occurs one)
[misclassified data](mnist/results/trial5_misclassification.png)
## On Dog vs. Cats dataset
Use VGG19 with weights pretrained ImageNet dataset as base model to extract features.
Train classifier with running,
```
$ python train_keras.py
```
It gets around 96.20% test accuracy after 27 epoch.

- Training curves
[training curves](dogs-vs-cats/results/trial1_training_curves.png)
### References
- [Confusion matrix — scikit-learn 0.21.3 documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)
- [keras/mnist_cnn.py at master · keras-team/keras](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
- [Image classification  |  TensorFlow Core](https://www.tensorflow.org/tutorials/images/classification)
- [Kerasのモデルは学習完了時のものが最良とは限らない - Qiita](https://qiita.com/cvusk/items/7bcd3bc2e82bb45c9e9c)
- [Dogs vs. Cats | Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)
- [Applications - Keras Documentation](https://keras.io/applications/#vgg19)
