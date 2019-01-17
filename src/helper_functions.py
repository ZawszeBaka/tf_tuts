import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import idx2numpy
import math

#--------------------------------
def get_dataset(show_info=False):
    '''
        Get datasets MNIST
    '''
    X_train = idx2numpy.convert_from_file('../datasets/mnist/X_train.idx')
    y_train = idx2numpy.convert_from_file('../datasets/mnist/y_train.idx')
    X_test = idx2numpy.convert_from_file('../datasets/mnist/X_test.idx')
    y_test = idx2numpy.convert_from_file('../datasets/mnist/y_test.idx')

    if show_info:
        print('[INFO] Train shape ', X_train.shape)
        print('[INFO] Test shape ', X_test.shape)

    return X_train, y_train, X_test, y_test

def show_some_images(X,y,predicted_y=None, num_images = 3):
    '''
        Args
            X : can be X_train or X_test
            y : can be y_train or y_test
    '''
    for i in range(num_images):
        plt.imshow(X[i])
        if predicted_y is not None:
            plt.title('Actual class : '+str(y[i])+' , Predicted class : '+str(predicted_y[i]))
        else:
            plt.title('Actual class : '+str(y[i]))
        plt.show()

def one_hot_encoding(y, num_classes, axis=1):
    '''
        Args
            y: can be y_train or y_test
            num_classes: number of classes
        Returns
            one_hot_matrix
    '''
    one_hot_matrix = tf.one_hot(y, num_classes, axis = axis)
    with tf.Session() as sess:
        one_hot_matrix = sess.run(one_hot_matrix)
    return one_hot_matrix

def flatting(X):
    '''
        Args
            X: can be X_train or X_test
        Returns

    '''
    return X.reshape(X.shape[0],-1)

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (number of examples, flat image size )
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0] # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def plot_cost(costs, learning_rate):
    '''
        Args:
            costs : list of cost each epoch
    '''
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('Learning rate = '+str(learning_rate))
    plt.show()

def calc_accuracy(y,predicted_y):
    '''
        Args
            
    '''

def plot_weights(ws,labels,sz=(40,40)):
    '''
        ws : weights
        labels : label (in text)
    '''
    for i in range(ws.shape[0]):
        plt.plot(w[i,:].imshow(sz))
        plt.title('Class ' + str(labels[i]))
        plt.show()
