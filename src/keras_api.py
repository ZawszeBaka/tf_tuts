import helper_functions
from helper_functions import *

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam

def main():

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



if __name__ == '__main__':
    main()
