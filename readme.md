# CNN Classifier with Keras

## TODO
- [ ] automatically output the experimental conditions to csv file eg. epochs, data amounts, 
- [ ] automatically set the prefix
- [ ] make callback for changing learning rate
- [x] comment each step
- [ ] make a generator for other dataset which has more amount
- [ ] use ImageGenerator to process image data before training
- [x] visualize confusion matrix

### Requirements
- macOS High Sierra 10.13.6
- Python 3.6.0
- Keras 2.3.0
- tensorflow 1.14.0 (backend)

In your pyenv environment, run 
```
$ pip install -r requirements.txt
```

## On MNIST dataset


## Dog vs. Cats dataset


### References
- [Confusion matrix — scikit-learn 0.21.3 documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)
- [keras/mnist_cnn.py at master · keras-team/keras](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
- [Image classification  |  TensorFlow Core](https://www.tensorflow.org/tutorials/images/classification)
- [Kerasのモデルは学習完了時のものが最良とは限らない - Qiita](https://qiita.com/cvusk/items/7bcd3bc2e82bb45c9e9c)