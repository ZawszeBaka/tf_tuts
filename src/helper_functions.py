import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import idx2numpy
import math
import pickle

#--------------------------------
print('[INFO] Tensorflow version : ' , tf.__version__)
print('[INFO] OpenCV version : ', cv2.__version__,end ='\n\n')

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

def show_some_images(X,y,predicted_y=None, subplot=(3,3)):
    '''
        Args
            X : can be X_train or X_test
            y : can be y_train or y_test
            subplot: 3 x 3
    '''

    fig, axes = plt.subplots(subplot[0],subplot[1])
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i],cmap='binary')

        if predicted_y is not None:
            xlabel = 'Actual:'+str(y[i])+',Predicted:'+str(predicted_y[i])
        else:
            xlabel = 'Actual:'+str(y[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])

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

def calc_accuracy(y,predicted_y,
                  show_false_predictions = True,  # if show false prediction
                  num_images = 3,
                  X = None):
    '''
        Args:
            y: raw y
            predicted_y: raw predicted y
        Returns
            accuracy
    '''
    equal_inds = np.equal(y,predicted_y)
    accuracy = y[equal_inds].shape[0] / y.shape[0]
    if show_false_predictions:
        if X is None:
            print('[ERROR] X is None !')
            return 0.0
        # get false predictions
        equal_inds = (equal_inds == False)
        plot_y = y[equal_inds]
        plot_predicted_y = predicted_y[equal_inds]
        plot_X = X[equal_inds]
        show_some_images(plot_X,plot_y,plot_predicted_y, subplot=(3,3))
    return accuracy

def plot_weights(ws,labels,size=(40,40),subplot=(3,4)):
    '''
        ws : weights  [ [weight1],
                        [weight2],...]
        labels : label (in text)
        subplot : 3 x 4
    '''
    # Get the lowest and highest values for the weights
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other
    w_min = np.min(ws)
    w_max = np.max(ws)

    # Create figure with 3x4 sub-plots
    # where the last 2 sub-plots are unused
    fig, axes = plt.subplots(subplot[0],subplot[1])
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots
        if i < labels.shape[0]:
            img = ws[i,:].reshape(size)

            # set the label for the sub-plot
            ax.set_xlabel('Weight {0}'.format(labels[i]))

            # plot the image
            ax.imshow(img, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot
        ax.set_xticks([])
        ax.set_yticks([])

    # show
    plt.show()

def save_parameters(file_name,parameters):
    print('[INFO] Saved to file ',file_name)
    with open('../model'+'/'+file_name,'wb') as f:
        pickle.dump(parameters, f)

def load_parameters(file_name):
    with open('../model'+'/'+file_name,'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    print('[WARNING] Do not run this file! Just import this file and use !')
