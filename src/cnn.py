import helper_functions
from helper_functions import *

if __name__ == '__main__':

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
    X_train = X_train_raw.reshape(list(X_train_raw.shape)+[1])   # after flatting
    Y_train = one_hot_encoding(Y_train_raw, num_classes) # after one hot encoding
    X_test = X_test_raw.reshape(list(X_test_raw.shape)+[1])   # after flatting
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
    img_shape = X_train.shape[1:]

    # Number of colour channels for the images: 1 channel for gray-scale
    if len(X_train.shape) == 3:
        num_channels = 1
    else:
        num_channels = X_train.shape[-1]

    learning_rate = 1e-4
    print('[INFO] Learning rate ', learning_rate)

    train_batch_size = 64
    print('[INFO] Train batch size = ', train_batch_size)

    num_iterations = 1
    print('[INFO] Num iterations = ',num_iterations)

    seed = 100
    print('[INFO] seed = ', seed)

    ################################################################

    x = tf.placeholder(tf.float32,
                       shape=[None, img_size, img_size, num_channels],
                       name='x')
    y = tf.placeholder(tf.float32,
                       shape=[None,num_classes],
                       name='y')
    actual_y = tf.argmax(y, axis=1)

    # Convolutional Layer 1
    layer_conv1, weights_conv1 = new_conv_layer(input=x,
                                                num_input_channels=num_channels,
                                                filter_size=filter_size1,
                                                num_filters=num_filters1,
                                                use_pooling=True)
    print('[INFO] Convolutional Layer 1', layer_conv1)

    # Convolutional Layer 2
    layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                                num_input_channels=num_filters1,
                                                filter_size=filter_size2,
                                                num_filters=num_filters2,
                                                use_pooling=True)
    print('[INFO] Convolutional Layer 2', layer_conv2)

    # Flatten Layer
    layer_flat, num_features = flatten_layer(layer_conv2)
    print('[INFO] Flattened layer', layer_flat)
    print('[INFO] Num features',num_features)

    # Fully-connected Layer 1
    layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size,
                             use_relu=True)
    print('[INFO] Fully-connected Layer 1', layer_fc1)

    # Fully-connected Layer 2
    layer_fc2 = new_fc_layer(input=layer_fc1,
                            num_inputs = fc_size,
                            num_outputs=num_classes,
                            use_relu=False)
    print('[INFO] Fully-connected Layer 2', layer_fc2)

    # Predicted Class
    predicted_y = tf.argmax(tf.nn.softmax(layer_fc2), axis=1)

    # Cost-function to be optimized
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                               labels=y)
    cost = tf.reduce_mean(cross_entropy)

    # Optimization Method
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Performance measures
    correct_prediction = tf.equal(actual_y, predicted_y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #################################################################
    # Tensorflow RUN
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Start-time
    start_time = time.time()

    costs = []
    for i in range(num_iterations):
        epoch_cost = 0
        mini_batches = random_mini_batches(X_train,
                                           Y_train,
                                           mini_batch_size=train_batch_size,
                                           seed = seed,
                                           is_flattened=False)
        for mini_batch in mini_batches:
            (x_batch, y_batch) = mini_batch
            _, mini_batch_cost = sess.run([optimizer, cost], feed_dict={x:x_batch,y:y_batch})
            epoch_cost += mini_batch_cost

        costs.append(epoch_cost)

        # if i%100 == 0:
        #     # Calculate the accuracy on the training-set
        #     acc = sess.run(accuracy, feed_dict={x:X_train,y:Y_train})
        #
        #     # Message
        #     msg = '[INFO] Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}, Cost: {1:>6.1%}'
        #     print(msg.format(i+1,acc,epoch_cost))

    # Ending time
    time_dif = time.time() - start_time

    # Difference between start and end-times
    print('Time usage: ' + str(timedelta(seconds=int(round(time_dif)))))

    sess.close()
