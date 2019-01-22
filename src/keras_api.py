import helper_functions
from helper_functions import *

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras import backend as K

def main(is_train = True, method='sequential', output_method = 1):
    '''
        is_train :  True , it will train the model
                    False, it will just load the pre-trained model
        method : 'sequential' or 'functional'
                 method of training process
        output_method : 1 or 2
    '''

    # Convolutional Layer 1
    filter_size1 = 5 # 5x5
    num_filters1 = 16 # 16 filters
    # Convolutional Layer 2
    filter_size2 = 5 # 5x5
    num_filters2 = 36 # 36 filters
    # Fully-connected layer
    fc_size = 128 # number of neurons in fully-connected layer

    # Load data
    X_train_raw, Y_train_raw, X_test_raw, Y_test_raw = get_dataset()
    num_classes = np.unique(Y_train_raw).shape[0]

    print('[INFO] Train X shape ', X_train_raw.shape)
    print('[INFO] Train Y shape', Y_train_raw.shape)
    print('[INFO] Test X shape ', X_test_raw.shape)
    print('[INFO] Test Y shape', Y_test_raw.shape)
    print('[INFO] Num classes ', num_classes)

    # Convert to one-hot
    X_train = flatting(X_train_raw)   # after flatting
    Y_train = one_hot_encoding(Y_train_raw, num_classes) # after one hot encoding
    X_test = flatting(X_test_raw)  # after flatting
    Y_test = one_hot_encoding(Y_test_raw, num_classes) # after one hot encoding

    print('[INFO] X train', X_train.shape)
    print('[INFO] Y train ', Y_train.shape)
    print('[INFO] X test', X_test.shape)
    print('[INFO] Y test ', Y_test.shape)

    # num_rows * num_cols
    flat_img_size = X_train_raw.shape[1] * X_train_raw.shape[2]

    # assume that the image has the same rows and cols
    img_size = X_train_raw.shape[1]

    # Tuple
    img_shape = list(X_train_raw.shape[1:])+[1]

    # Number of colour channels for the images: 1 channel for gray-scale
    if len(X_train.shape) == 3:
        num_channels = 1
    else:
        num_channels = X_train.shape[-1]

    learning_rate = 1e-4
    print('[INFO] Learning rate ', learning_rate)

    train_batch_size = 64
    print('[INFO] Train batch size = ', train_batch_size)

    num_epochs = 1
    print('[INFO] Num epochs = ',num_epochs)

    seed = 100
    print('[INFO] seed = ', seed)

    if is_train:
        if method == 'sequential':
            ################################################################
            # Start construction of the Keras Sequential model
            model = Sequential()

            # Add an input layer which is similar to a feed_dict in Tensorflow
            # Note that the input-shape must be a tuple containing the image-size
            model.add(InputLayer(input_shape = (flat_img_size,)))

            # The input is a flattened array with 784 elements,
            # but the convolutional layers expects images with shape (28,28)
            model.add(Reshape(img_shape))

            # First convolutional layer with ReLU-activation and max-pooling
            model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                             activation='relu', name='layer_conv1'))
            model.add(MaxPooling2D(pool_size=2, strides=2))

            # Second convolutional layer with ReLU-activation and max-pooling
            model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                             activation='relu', name='layer_conv2'))
            model.add(MaxPooling2D(pool_size=2, strides=2))

            # Flatten the 4-rank ouput of the convolutional layers
            # to 2-rank that can be input to a fully-connected / dense layer
            model.add(Flatten())

            # First fully-connected / dense layer with ReLU-activation
            model.add(Dense(128, activation='relu'))

            # Last fully-connected / dense layer with softmax-activation
            # for use in classification
            model.add(Dense(num_classes, activation='softmax'))

            # Model Compilation
            optimizer = Adam(lr=learning_rate) # lr : learning_rate

            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # Training
            model.fit(x=X_train,
                      y=Y_train,
                      epochs=num_epochs,
                      batch_size=train_batch_size)

            # Evaluation
            rs = model.evaluate(x=X_test,
                                y=Y_test)

            for name, value in zip(model.metrics_names, rs):
                print('[INFO]',name, value)

            # Prediction
            predicted_y_test = model.predict(x=X_test)
            predicted_y_test = np.argmax(predicted_y_test,axis=1)
            print('[DEBUG] The first 9 predicted y test : ', predicted_y_test[:9] )
            show_some_images(X=X_test_raw,y=Y_test_raw,predicted_y=predicted_y_test)

        elif method=='functional':
            # Create an input layer which is similar to a feed_dict in Tensorflow
            # Note that the input-shape must be a tuple containing the image-size
            inputs = Input(shape=(flat_img_size,))

            # Variable used for building the Neural Network
            net = inputs

            # The input is an image as a flattened array with 784 elements
            # But the convolutional layers expect images with shape (28,28,1)
            net = Reshape(img_shape)(net)

            # First convolutional layer with ReLU-activation and max-pooling
            net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                         activation='relu', name='layer_conv1')(net)
            net = MaxPooling2D(pool_size=2, strides=2)(net)

            # Second convolutional layer with ReLU-activation and max-pooling
            net = Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                         activation='relu', name='layer_conv2')(net)
            net = MaxPooling2D(pool_size=2, strides=2)(net)

            # Flatten the output of the conv-layer from 4-dim to 2-dim
            net = Flatten()(net)

            # First fully-connected / dense layer with ReLU-activation
            net = Dense(128, activation='relu')(net)

            # Last fully-connected / dense layer with softmax-activation
            # so it can be used for classification
            net = Dense(num_classes, activation='softmax')(net)

            # Output of the Neural Network
            outputs = net

            # Model Compilation
            optimizer = Adam(lr=learning_rate) # lr : learning_rate
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=optimizer,  #'rmsprop', # RMSprop optimizer
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # Training
            model.fit(x=X_train,y=Y_train,
                      epochs=num_epochs, batch_size=train_batch_size)

            # Evaluation
            rs = model.evaluate(x=X_test,
                                y=Y_test)

            for name, value in zip(model.metrics_names, rs):
                print('[INFO]', name, value)

            # Prediction
            predicted_y = model.predict(x=X_test)
            predicted_y = np.argmax(predicted_y, axis=1)
            show_some_images(X=X_test_raw,y=Y_test_raw,predicted_y=predicted_y)

        path_model = '../model/keras_api.keras'
        print('[INFO] Save model to ', path_model)
        model.save(path_model)

        print(model.summary())

        # delete the model from memory so we are sure it is no longer used
        del model

        # Load model from file
        model = load_model(path_model)

    else:
        path_model = '../model/keras_api.keras'
        print('[INFO] Load model from ', path_model)
        # Load model from file
        model = load_model(path_model)

        # Evaluation
        rs = model.evaluate(x=X_test,
                            y=Y_test)

        for name, value in zip(model.metrics_names, rs):
            print('[INFO]', name, value)

    print(model.summary())

    # We count the indices to get the layers we want
    layer_input = model.layers[0] # input-layer
    layer_conv1 = model.layers[2] # first conv-layer
    layer_conv2 = model.layers[4] # second conv-layer

    # Convolutional Weights
    weights_conv1 = layer_conv1.get_weights()[0]
    print('[INFO] Weights (conv-layer1) shape :',weights_conv1.shape)
    plot_conv_weights(weights_conv1, input_channel=0)

    weights_conv2 = layer_conv2.get_weights()[0]
    print('[INFO] Weights (conv-layer2) shape :',weights_conv2.shape)
    plot_conv_weights(weights_conv2, input_channel=0)

    if output_method == 1:
        ## Output of convolutional layer - Method 1
        output_conv1 = K.function(inputs=[layer_input.input],
                                  outputs=[layer_conv1.output])
        output_conv2 = K.function(inputs=[layer_input.input],
                                  outputs=[layer_conv2.output])

        layer_output1 = output_conv1([[X_test_raw[0,:,:]]])[0]
        print('[INFO] Output of conv-layer 1 shape ',layer_output1.shape)
        plot_conv_output(values = layer_output1)
    elif output_method == 2:
        ## Output of convolutional layer - Method 2
        output_conv2 = Model(inputs = layer_input.input,
                             outputs = layer_conv2.output)
        layer_output2 = output_conv2.predict(np.array([X_test_raw[0,:,:]]))
        print('[INFO] Output of conv-layer 2 shape ',layer_output2.shape)
        plot_conv_output(values=layer_output2)







if __name__ == '__main__':
    main(is_train=True,method='functional',output_method=1)
