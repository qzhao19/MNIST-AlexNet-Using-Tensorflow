# MNIST-AlexNet-Using-Tensorflow
Using AlexNet neural networks for the classic dataset MNIST

# Overview

MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. This dataset is made up of images of handwritten digits, 28x28 pixels in size. 

In this repertoire, I created AlexNet network in tensorflow to perform on the mnist dataset. AlexNet is a helpful deep convolutional network for image recognition task, which significantly outperformed all the prior competitors and won the challenge ILSVRC 2012 by reducing the top-5 error from 26% to 15.3\%. The network architecture presents image below.

![image](https://github.com/zhaoqi19/MNIST-AlexNet-Using-Tensorflow/blob/master/image/AlexNet.png)

For more details on the underlying model please refer to the following paper:

    @incollection{NIPS2012_4824,
      title = {ImageNet Classification with Deep Convolutional Neural Networks},
      author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
      booktitle = {Advances in Neural Information Processing Systems 25},
      pages = {1097--1105},
      year = {2012}
    }


# Requirement

- Python 3.6
- Tensorflow >= 1.14
- matplotlib 3.1.1

# Contents

- `model.py`: Class with the graph definition of the AlexNet.
- `layers.py`: Neuron network layers containing convolutional layer, full collection layer, normalization and maxpooling.
- `evals.py`: Model evaluation's function.
- `train.py`: Script to run the training process.
- `images/*`: contains three example images.
- `outputs`: output result folder containing two sub-folder (accuracy_loss and model)

# Usages

First, I strongly recommend to take a look at the entire code of this repository. In fact, even Tensorflow and Keras allow us to import and download the MNIST dataset directly from their API. Therefore, I will start with the following two lines to import tensorflow and MNIST dataset under the Tensorflow API. A local training job can be run with the following command:

    python train.py \
        --valid_steps=11 \
        --max_steps=1001 \
        --batch_size=128 \
        --base_learning_rate=0.001 \
        --input_shape=784 \
        --num_classes=10 \
        --keep_prob=0.75 \
        --save_dir='./outputs' \
        --tb_path='./tb_logs' 




  


