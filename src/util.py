import sys
import gzip
import os
import time
import matplotlib.pyplot as plt
import pickle
import numpy as np
from pandas_ml import ConfusionMatrix

#from Lasagne's MNIST.py
def load_mnist_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading fashionMNIST images and labels.
    # For convenience, they also download the requested files if needed.
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version

        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_test, y_test


def get_log_path():
    log_basedir = './graphs'
    run_label = time.strftime('%d-%m-%Y_%H-%M-%S')  # e.g. 12-11-2016_18-20-45
    return os.path.join(log_basedir, run_label)

def plot9images(images, cls_true, img_shape, cls_pred=None, lspace=0.3):
    """
    Function to show 9 images with their respective classes.
    If cls_pred is an array, you can see the image and the prediction.

    :type images: np array
    :type cls_true: np array
    :type img_shape: np array
    :type cls_prediction: None or np array
    """
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=lspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0} Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def plot15images(images, cls_true, img_shape, cls_pred=None):
    """
    Function to show 15 images with their respective classes.
    If cls_pred is an array, you can see the image and the prediction.

    :type images: np array
    :type cls_true: np array
    :type img_shape: np array
    :type cls_prediction: None or np array
    """
    assert len(images) == len(cls_true) == 15
    fig, axes = plt.subplots(3, 5, figsize=(11, 11))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plotconfusion(truth, predictions):
    """
    Function to plot the confusion fuction between the
    truth and predictions array.

    :type truth: np array
    :type predictions: np array
    """
    cm = ConfusionMatrix(truth, predictions)
    _ = plt.figure(figsize=(10, 10))
    _ = cm.plot()
    _ = plt.show()


#return fashionMNIST label names for an array 
label_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
def get_label_names(pred):
    return [label_names[k] for k in pred]


#missed é um array com True onde errou, n o numero de exemplos que queremos
#devolve o índice dos exemplos
def select_missed_examples(missed, n):
    #ind são os indices de 0 a size
    ind = np.arange(missed.size)
    #ind são os indices com True
    ind = ind[missed]
    np.random.shuffle(ind)
    return ind[0:n]

def plot_missed_examples(images, truth, missed, predicted=None):
    image_shape = (images.shape[1], images.shape[2])
    missing = select_missed_examples(missed, 9)
    if predicted is not None:
        plot9images(images[missing], get_label_names(truth[missing]), image_shape, get_label_names(predicted[missing]), lspace=0.9)
    else:
        plot9images(images[missing], get_label_names(truth[missing]), image_shape)
    
