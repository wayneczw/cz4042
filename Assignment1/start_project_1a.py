#
# Project 1, starter code part a
#
# import matplotlib
# matplotlib.use('Agg')
import math
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import multiprocessing as mp
import time


NUM_FEATURES = 36
NUM_CLASSES = 6
epochs = 1000
seed = 10
np.random.seed(seed)


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)
#end def


def train(trainX, trainY, testX, testY, small=False, num_hidden_layer=1, batch_size=32, num_neurons=10, learning_rate=0.01, l2_beta=10**(-6), **kwargs):
    def _train_1():
        # create the model
        x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
        y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

        # Build the graph for the deep net
        W = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neurons], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
        B  = tf.Variable(tf.zeros([num_neurons]), name='biases')
        Z  = tf.matmul(x, W) + B  #synaptic input to hidden-layer
        H = tf.nn.sigmoid(Z)

        V = tf.Variable(tf.truncated_normal([num_neurons, NUM_CLASSES], stddev=1.0/math.sqrt(float(num_neurons))), name='weights')
        C = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        U  = tf.matmul(H, V) + C  #synaptic input to output-layer

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=U)
        regularization = tf.nn.l2_loss(W) + tf.nn.l2_loss(V)
        loss = tf.reduce_mean(cross_entropy + l2_beta*regularization)

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        correct_prediction = tf.cast(tf.equal(tf.argmax(U, 1), tf.argmax(y_, 1)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        
        train_err = []
        test_acc = []
        N = len(trainX)
        idx = np.arange(N)
        start_time = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                np.random.shuffle(idx)
                trainXX = trainX[idx]
                trainYY = trainY[idx]

                for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                    train_op.run(feed_dict={x: trainXX[start:end], y_: trainYY[start:end]})
                
                train_err.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
                test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
                if i % 100 == 0:
                    print('batch_size %g: iter %d: train error %g'%(batch_size, i, train_err[i]))
                    print('batch_size %g: iter %d: test accuracy %g'%(batch_size, i, test_acc[i]))
                    print('----------------------')
            #end for
        #end with

        time_taken = time.time() - start_time

        return train_err, test_acc, time_taken
    #end def

    def _train_2():
        # create the model
        x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
        y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

        # Build the graph for the deep net
        W = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neurons], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
        B = tf.Variable(tf.zeros([num_neurons]), name='biases')
        Z = tf.matmul(x, W) + B  #synaptic input to hidden-layer
        H = tf.nn.sigmoid(Z)

        V = tf.Variable(tf.truncated_normal([num_neurons, num_neurons], stddev=1.0/math.sqrt(float(num_neurons))), name='weights')
        C = tf.Variable(tf.zeros([num_neurons]), name='biases')
        U = tf.matmul(H, V) + C  #synaptic input to output-layer
        G = tf.nn.relu(U)

        R = tf.Variable(tf.truncated_normal([num_neurons, NUM_CLASSES], stddev=1.0/math.sqrt(float(num_neurons))), name='weights')
        D = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        O = tf.matmul(G, R) + D  #synaptic input to output-layer
        K = tf.nn.relu(O)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=K)
        regularization = tf.nn.l2_loss(W) + tf.nn.l2_loss(V) + tf.nn.l2_loss(R)
        loss = tf.reduce_mean(cross_entropy + l2_beta*regularization)

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        correct_prediction = tf.cast(tf.equal(tf.argmax(U, 1), tf.argmax(y_, 1)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        
        train_err = []
        test_acc = []
        N = len(trainX)
        idx = np.arange(N)
        start_time = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                np.random.shuffle(idx)
                trainXX = trainX[idx]
                trainYY = trainY[idx]

                for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                    train_op.run(feed_dict={x: trainXX[start:end], y_: trainYY[start:end]})
                
                train_err.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
                test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
                if i % 100 == 0:
                    print('batch_size %g: iter %d: train error %g'%(batch_size, i, train_err[i]))
                    print('batch_size %g: iter %d: test accuracy %g'%(batch_size, i, test_acc[i]))
                    print('----------------------')
            #end for
        #end with

        time_taken = time.time() - start_time

        return train_err, test_acc, time_taken
    #end def

    if small:  # experiment with small datasets
        trainX = trainX[:100]
        trainY = trainY[:100]

    if num_hidden_layer == 1:
        train_err, test_acc, time_taken = _train_1()
    elif num_hidden_layer == 2:
        train_err, test_acc, time_taken = _train_2()

    return train_err, test_acc, time_taken
#end def


def _read_data(file_name, x_min=None, x_max=None, train=False):
    _input = np.loadtxt(file_name, delimiter=' ')
    X, _Y = _input[:, :36], _input[:, -1].astype(int)
    if train:
        x_min = np.min(X, axis=0)
        x_max = np.max(X, axis=0)

    X = scale(X, x_min, x_max)
    _Y[_Y == 7] = 6

    # one hot matrix
    Y = np.zeros((_Y.shape[0], NUM_CLASSES))
    Y[np.arange(_Y.shape[0]), _Y-1] = 1
    if train: return X, Y, x_min, x_max
    else: return X, Y
#end def


def main():
    # read train data
    trainX, trainY, x_min, x_max = _read_data('sat_train.txt', train=True)

    # read test data
    testX, testY = _read_data('sat_test.txt', x_min=x_min, x_max=x_max)

    train_test = dict(trainX=trainX, trainY=trainY, testX=testX, testY=testY)

    ############## Q 1 ###########
    train_err, test_acc, train_time = train(**train_test)
    plt.figure('1 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    plt.plot(range(epochs), train_err)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Train Error')
    plt.show()

    ############## Q 2 ###########
    batch_sizes = [4, 8, 16, 32, 64]
    train_err = []
    test_acc = []
    train_time = []
    for batch_size in batch_sizes:
        _train_err, _test_acc, _train_time = train(batch_size=batch_size, **train_test)
        train_err.append(_train_err)
        test_acc.append(_test_acc)
        train_time.append(_train_time)

    # plot train_err against epoch for each batch size
    # plot test_acc against epoch for each batch size
    for i in range(len(batch_sizes)):
        # train_err
        plt.figure("Batch Size {}: Train Error against Epoch".format(batch_sizes[i]))
        plt.plot(range(epochs), train_err[i])
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Train Error')
        plt.show()

        # test_acc
        plt.figure("Batch Size {}: Test Accuracy against Epoch".format(batch_sizes[i]))
        plt.plot(range(epochs), test_acc[i])
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Test Accuracy')
        plt.show()
    #end for

    # plot training time taken for each batch size
    plt.figure('Training Time against Batch_size')
    plt.plot(batch_sizes, train_time)
    plt.xlabel('Batch Size')
    plt.ylabel('Time/Seconds')
    plt.show()

    # plot test accuracy against batch size
    final_acc = [acc[-1] for acc in test_acc]
    plt.figure('Accuracy against Batch Size')
    plt.plot(batch_sizes, final_acc)
    plt.xlabel('Batch Size')
    plt.ylabel('Test accuracy')
    plt.show()

    best_batch_size = 32

    ############## Q 3 ###########
    num_neurons_list = [5, 10, 15, 20, 25]
    train_err = []
    test_acc = []
    train_time = []
    for num_neurons in num_neurons_list:
        _train_err, _test_acc, _train_time = train(num_neurons=num_neurons, batch_size=best_batch_size, **train_test)
        train_err.append(_train_err)
        test_acc.append(_test_acc)
        train_time.append(_train_time)

    # plot train_err against epoch for each Number of Neurons
    # plot test_acc against epoch for each Number of Neurons
    for i in range(len(num_neurons_list)):
        # train_err
        plt.figure("Number of Neurons {}: Train Error against Epoch".format(num_neurons_list[i]))
        plt.plot(range(epochs), train_err[i])
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Train Error')
        plt.show()

        # test_acc
        plt.figure("Number of Neurons {}: Test Accuracy against Epoch".format(num_neurons_list[i]))
        plt.plot(range(epochs), test_acc[i])
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Test Accuracy')
        plt.show()
    #end for

    # plot training time taken for each Number of Neurons
    plt.figure('Training Time against Number of Neurons')
    plt.plot(num_neurons_list, train_time)
    plt.xlabel('Number of Neurons')
    plt.ylabel('Time/Seconds')
    plt.show()

    # plot test accuracy against Number of Neurons
    final_acc = [acc[-1] for acc in test_acc]
    plt.figure('Accuracy against Number of Neurons')
    plt.plot(num_neurons_list, final_acc)
    plt.xlabel('Number of Neurons')
    plt.ylabel('Test Accuracy')
    plt.show()
    best_num_neurons = 10

    ############## Q 4 ###########
    l2_beta_list = [0, 10**(-3), 10**(-6), 10**(-9), 10**(-12)]
    train_err = []
    test_acc = []
    train_time = []
    for l2_beta in l2_beta_list:
        _train_err, _test_acc, _train_time = train(l2_beta=l2_beta, batch_size=best_batch_size, num_neurons=best_num_neurons, **train_test)
        train_err.append(_train_err)
        test_acc.append(_test_acc)
        train_time.append(_train_time)

    # plot train_err against epoch for each batch size
    # plot test_acc against epoch for each batch size
    for i in range(len(l2_beta_list)):
        # train_err
        plt.figure("Decay Parameter {}: Train Error against Epoch".format(l2_beta_list[i]))
        plt.plot(range(epochs), train_err[i])
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Train Error')
        plt.show()

        # test_acc
        plt.figure("Decay Parameter {}: Test Accuracy against Epoch".format(l2_beta_list[i]))
        plt.plot(range(epochs), test_acc[i])
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Test Accuracy')
        plt.show()
    #end for

    # plot training time taken for each Decay Parameter
    plt.figure('Training Time against Decay Parameter')
    plt.plot(l2_beta_list, train_time)
    plt.xlabel('Decay Parameter')
    plt.ylabel('Time/Seconds')
    plt.show()

    # plot test accuracy against Decay Parameter
    final_acc = [acc[-1] for acc in test_acc]
    plt.figure('Accuracy against Decay Parameter')
    plt.plot(l2_beta_list, final_acc)
    plt.xlabel('Decay Parameter')
    plt.ylabel('Test Accuracy')
    plt.show()
    # best_l2_beta = 

    ############## Q 5 ###########
    train_err = []
    test_acc = []
    train_time = []
    _train_err, _test_acc, _train_time = train(num_hidden_layer=2, **train_test)
    train_err.append(_train_err)
    test_acc.append(_test_acc)
    train_time.append(_train_time)

    # plot train_err against epoch
    plt.figure('Training Error: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    plt.plot(range(epochs), train_err)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Train Accuracy')
    plt.show()

    # plot test_acc against epoch
    plt.figure('Test Accuracy: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    plt.plot(range(epochs), test_acc[i])
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Test Accuracy')
    plt.show()
#end def


if __name__ == '__main__': main()

