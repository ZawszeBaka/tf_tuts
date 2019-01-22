import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import idx2numpy
import math
import pickle
import time
from datetime import timedelta

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

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0, is_flattened=True):
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
    if is_flattened:
        shuffled_X = X[permutation,:]
        shuffled_Y = Y[permutation,:]
    else:
        shuffled_X = X[permutation,:,:]
        shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        if is_flattened:
            mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
            mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        else:
            mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:]
            mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        if is_flattened:
            mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        else:
            mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:]
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
            img = ws[:,i].reshape(size)

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

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05, seed=100))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,               # the previous layer
                   num_input_channels,  # Num. channels in previous layer
                   filter_size,         # Width, height of each filter
                   num_filters,         # Number of filters
                   use_pooling=True):   # Use 2x2 max-pooling
    # Shape of the filter-weights for the convolution
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka filters with the given shape
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolutionself.
    # Note: the strides are set to 1 in all dimensions
    # The first and last stride must always be 1, because
    # the first is for the image-number and
    # the last is for the input-channel
    # But e.g. strides = [1,2,2,1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image
    # The padding is set to 'SAME' which means the input imag
    # is padded with zeroes so the size of the ouput is the same
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1,1,1,1], # stride 1 pixel across the x- and y-axis
                         padding='SAME') # add zero-padding

    # Add the biases to the results of the convolution
    # A bias-value is added to each filter-channel
    layer += biases

    # Use pooling to down-sample the image resolution ?
    if use_pooling:
        # This is 2x2 max-pooling, which means that
        # we consider 2x2 windows and select
        # the largest value in each window. Then
        # we move 2 pixels to the next window
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1,2,2,1],
                               strides=[1,2,2,1], # move 2 pixels to the next window
                               padding='SAME')

    # Rectified Linear Unit (ReLU)
    # It calculates max(x, 0) for each input pixel x
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first

    return layer,weights

def flatten_layer(layer):
    '''
    Input:
        The shape of the input layer is assumed to be:
        layer_shape == [num_images, img_height, img_width, num_channels]

    Returns
        flattened layer which has the shape:
        [num_images, img_height*img_width*num_channels]
    '''
    # Get the shape of the input layer
    layer_shape = layer.get_shape()

    # The number of features is: img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements() # tensorflow api

    # Reshape the layer to [num_images, num_features]
    # Note that we just set the size of the second dimension to
    # num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping
    layer_flat = tf.reshape(layer, [-1,num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height*img_width*num_channels]

    return layer_flat, num_features

def new_fc_layer(input,             # The previous layer
                 num_inputs,        # Num. inputs from previous layer
                 num_outputs,       # Num. outputs
                 use_relu=True):    # Use Rectified Linear Unit (ReLU)?
    # Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values
    layer = tf.add(tf.matmul(input, weights),biases)

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def plot_confusion_matrix(y,predicted_y,num_classes):
    '''
    Args
        y : true class
        predicted_y : predicted class
    '''
    # Get the confusion matrix using sklearn
    cm = confusion_matrix(y_true=y,y_pred=predicted_y)

    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted class')
    plt.ylabel('True class')

    plt.show()

def plot_conv_weights(w, input_channel=0):
    '''
    Assume weights are Tensorflow ops for 4-dim variables
    e.g. weights_conv1 or weights_conv2
    '''
    # Get the lowest and highest values
    # This is used to correct the color intensity across
    # the images so they can be compared with each other
    w_min = np.min(w)
    w_max = np.max(w)

    # w.shape = [filter_size, filter_size, num_input_channels, num_filters ]
    num_filters = w.shape[3]

    # Number of grids to plot
    # Rounded-up, square-root of the number of filters
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights
    for i, ax in enumerate(axes.flat):
        # only plot the valid filter-weights
        if i < num_filters:
            img = w[:,:,input_channel,i]
            ax.imshow(img,vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def plot_conv_layer(layer,image,title=''):
    '''
    Arg
        layer : Tensorflow op that outputs a 4-dim tensor
        which is the output of a convolutional layer,
        e.g. layer_conv1 or layer_conv2
        is the result of sess.run(layer, feed_dict = {x:[image]})
        has the shape [num_images, img_height, img_width, num_channels]
    '''
    # Num of filters used in the convolutional layer
    # [num_images, img_height, img_width, num_channels]
    num_filters = layer.shape[3]

    # Number of grids to plot
    # Rounded-up, square-root of the number of filters
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            # [num_images, img_height, img_width, num_channels]
            img = layer[0,:,:,i]
            ax.imshow(img, interpolation='neareast', cmap='binary')

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=16)

    plt.show()

def plot_conv_output(values):
    '''
        output_conv1 = K.function(inputs=[layer_input.input],
                                  outputs=[layer_conv1.output])
        layer_output1 = output_conv1([[X_test_raw[0,:,:]]])[0]
    '''
    # Number of filters used in the conv.layer
    num_filters = values.shape[3]

    # Number of grids to plot
    # Rounded-up, square-root of the number of filters
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots
    fig, axes = plt.subplots(num_grids,num_grids)

    # Plot the output images of all the filters
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            # channel, height, width, image indices
            img = values[0,:,:,i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()







if __name__ == '__main__':
    print('[WARNING] Do not run this file! Just import this file and use !')
