import helper_functions
from helper_functions import *

'''

    1 layer

    Input:
        "None" is number of training samples
        X : [None, flat_img_size]
        w : [flat_img_size, num_classes]
        Y : [None, num_classes]


                X                 w             Y
                            [ .. .. .. ]
        [ .. , .. , .. ] X  [ .. .. .. ] + [ .. , .. , .. ]
                            [ .. .. .. ]

'''

if __name__ == '__main__':

    # Get dataset Mnist
    X_train_raw, Y_train_raw, X_test_raw, Y_test_raw = get_dataset()
    num_classes = np.unique(Y_train_raw).shape[0]

    print('[INFO] Train X shape ', X_train_raw.shape)
    print('[INFO] Train Y shape', Y_train_raw.shape)
    print('[INFO] Test X shape ', X_test_raw.shape)
    print('[INFO] Test Y shape', Y_test_raw.shape)
    print('[INFO] Num classes ', num_classes)

    # show_some_images(X_train_raw,Y_train_raw)

    # Convert to one-hot
    X_train = flatting(X_train_raw)   # after flatting
    Y_train = one_hot_encoding(Y_train_raw, num_classes) # after one hot encoding
    X_test = flatting(X_test_raw)   # after flatting
    Y_test = one_hot_encoding(Y_test_raw, num_classes) # after one hot encoding

    print('[INFO] X train', X_train.shape)
    print('[INFO] Y train ', Y_train.shape)
    print('[INFO] X test', X_test.shape)
    print('[INFO] Y test ', Y_test.shape)

    m = X_train.shape[0] # number of training set
    flat_img_size = X_train.shape[1] # flat image size

    seed = 10
    num_epochs = 10
    mini_batch_size = 100
    learning_rate = 0.0001
    print('[INFO] Seed = ', seed)
    print('[INFO] Num epochs = ', num_epochs)
    print('[INFO] Mini batch size = ', mini_batch_size)
    print('[INFO] Learning rate = ', learning_rate)

    # Creating placeholder variables
    X = tf.placeholder(tf.float32, shape=(None, flat_img_size), name='X') # type, shape
    W = tf.get_variable('W',[flat_img_size, num_classes],initializer = tf.contrib.layers.xavier_initializer(seed=seed))
    b = tf.get_variable('b',[num_classes],initializer = tf.zeros_initializer())
    Y = tf.placeholder(tf.float32, shape=(None, num_classes), name='Y')
    params = {'W':W, 'b':b}

    Z = tf.add(tf.matmul(X, W), b)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits = Z,
        labels = Y
    ))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    ## For predicting
    # # Calculate the correct predictions
    # correct_prediction = tf.equal(tf.argmax(Z),tf.argmax(Y))
    # # Calculate accuracy on the test set
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    predicted_Y = tf.argmax(Z, axis=1); # index of the largest element in each row

    init = tf.global_variables_initializer()

    costs=[]
    with tf.Session() as sess:
        print('[INFO] Initialize Variables ',end='')
        sess.run(init)
        print('Done!')
        print('\n[INFO] Training ... ')

        for epoch in range(num_epochs):
            epoch_cost = 0.0
            seed = seed + 1 # change the seed , make sure the random mini batches returns different pairs
            mini_batches = random_mini_batches(X_train, Y_train,
                                               mini_batch_size, seed)

            for mini_batch in mini_batches:
                (mini_batch_X, mini_batch_Y) = mini_batch

                # run the session to eXecute the "optimizer" and the "cost"
                # the feed_dict should contain a minibatch for X, Y
                _, mini_batch_cost = sess.run([optimizer,cost],
                                              feed_dict={X:mini_batch_X,
                                                         Y:mini_batch_Y})

                epoch_cost += mini_batch_cost / mini_batch_X.shape[0]

            if epoch % 100 == 0:
                print('[INFO] Cost after epoch %i : %.3f' % (epoch, epoch_cost))
            costs.append(epoch_cost)

        print('[INFO] Training .Done!\n')

        # save the params{'w':w, 'b':b}
        params = sess.run(params)
        model = dict()
        model['params'] = params
        model['costs'] = costs
        model['learning_rate'] = learning_rate
        model['num_epochs'] = num_epochs
        save_parameters('simple_linear_model.pickle',model)

        predicted_Y = sess.run(predicted_Y,
                               feed_dict={X:X_test,Y:Y_test})

    accuracy = calc_accuracy(Y_test_raw ,predicted_Y,
                            show_false_predictions=True,
                            X=X_test_raw)
    print('[INFO] Accuracy',accuracy)

    plot_weights(params['W'],np.unique(Y_test_raw),size=X_train_raw.shape[1:])

    plot_cost(costs,learning_rate)
