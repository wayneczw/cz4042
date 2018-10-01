#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
import time

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')


NUM_FEATURES = 36
NUM_CLASSES = 6

# learning_rate = 0.01
epochs = 1000
seed = 10
np.random.seed(seed)

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)
#end def

def train(trainX, trainY, testX, testY, small=False, num_hidden_layer=1, batch_size=32, num_neurons=10, learning_rate=0.01, beta=10**(-6), **kwargs):
    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    def ffn_1():
        # With 1 hidden layer
        # Build the graph for the deep net
        w1 = tf.Variable(
          tf.truncated_normal([NUM_FEATURES, num_neurons],
                              stddev=1.0 / math.sqrt(float(NUM_FEATURES))),
          name='weights')
        b1 = tf.Variable(tf.zeros([num_neurons]),name='biases')
        h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

        w2 = tf.Variable(
          tf.truncated_normal([num_neurons, NUM_CLASSES],
                              stddev=1.0 / math.sqrt(float(num_neurons))),
          name='weights')
        b2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        y = tf.matmul(h1, w2) + b2

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y_, logits=y)

        regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)

        loss = tf.reduce_mean(cross_entropy + beta*regularization)

        # Add a scalar summary for the snapshot loss.
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

        train_err = []
        test_acc = []
        time_to_update = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # time_to_update = 0

            no_data = len(trainX)
            idx = np.arange(no_data)
            for i in range(epochs):
                np.random.shuffle(idx)
                train_X, train_Y = trainX[idx], trainY[idx]

                t = time.time()
                for start, end in zip(range(0, no_data, batch_size), range(batch_size, no_data, batch_size)):
                    train_op.run(feed_dict={x: train_X[start:end], y_: train_Y[start:end]})
                time_to_update += (time.time() - t)
                train_err.append(loss.eval(feed_dict={x: train_X, y_: train_Y}))
                test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

                if i % 100 == 0:
                    print('batch size: %d: hidden neurons: [%d] decay parameters: %g iter: %d, train error  : %g'%(batch_size, num_neurons, beta, i, train_err[i]))
                    print('batch size: %d: hidden neurons: [%d] decay parameters: %g iter: %d, test accuracy  : %g'%(batch_size, num_neurons, beta, i, test_acc[i]))
                    print('-'*50)

        time_taken_one_epoch = (time_to_update/epochs) * 1000
        return train_err, test_acc, time_taken_one_epoch

    def ffn_2():
        # With 2 hidden layers
        # Build the graph for the deep net
        w1 = tf.Variable(
          tf.truncated_normal([NUM_FEATURES, num_neurons],
                              stddev=1.0 / math.sqrt(float(NUM_FEATURES))),
          name='weights')
        b1 = tf.Variable(tf.zeros([num_neurons]),name='biases')
        h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

        w2 = tf.Variable(
          tf.truncated_normal([num_neurons, num_neurons],
                              stddev=1.0 / math.sqrt(float(num_neurons))),
          name='weights')
        b2 = tf.Variable(tf.zeros([num_neurons]), name='biases')
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

        w3 = tf.Variable(
          tf.truncated_normal([num_neurons, NUM_CLASSES],
                              stddev=1.0 / math.sqrt(float(num_neurons))),
          name='weights')
        b3 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        y = tf.matmul(h2, w3) + b3

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)

        regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)

        loss = tf.reduce_mean(cross_entropy + beta*regularization)

        # Add a scalar summary for the snapshot loss.
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

        train_err = []
        test_acc = []
        time_to_update = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # time_to_update = 0

            no_data = len(trainX)
            idx = np.arange(no_data)
            for i in range(epochs):
                np.random.shuffle(idx)
                train_X, train_Y = trainX[idx], trainY[idx]

                t = time.time()
                for start, end in zip(range(0, no_data, batch_size), range(batch_size, no_data, batch_size)):
                    train_op.run(feed_dict={x: train_X[start:end], y_: train_Y[start:end]})
                time_to_update += time.time() - t
                train_err.append(loss.eval(feed_dict={x: train_X, y_: train_Y}))
                test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

                if i % 100 == 0:
                    print('batch size: %d: hidden neurons: [%d %d] decay parameters: %g iter: %d, train error  : %g'%(batch_size, num_neurons, num_neurons, beta, i, train_err[i]))
                    print('batch size: %d: hidden neurons: [%d %d] decay parameters: %g iter: %d, test accuracy  : %g'%(batch_size, num_neurons, num_neurons, beta, i, test_acc[i]))
                    print('-'*50)

        time_taken_one_epoch = (time_to_update/epochs) * 1000

        return train_err, test_acc, time_taken_one_epoch

    if small:
        # experiment with small datasets
        trainX = trainX[:1000]
        trainY = trainY[:1000]

    if num_hidden_layer == 1:
        train_err, test_acc, time_taken_one_epoch = ffn_1()

    elif num_hidden_layer == 2:
        train_err, test_acc, time_taken_one_epoch = ffn_2()

    return train_err, test_acc, time_taken_one_epoch
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
    # =====================Q1 Design a ffn with one hidden layer=====================

    train_err_ffn1, test_acc_ffn1, time_taken_one_epoch_ffn1 = train(**train_test)
    # plot train_err against epoch
    plt.figure('Training Error: 1 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    plt.title('Training Error')
    plt.plot(range(epochs), train_err_ffn1)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Train Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/1a_train_error_with_3_layer_network.png')

    # plot test_acc against epoch
    plt.figure('Test Accuracy: 1 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    plt.title('Test Accuracy')
    plt.plot(range(epochs), test_acc_ffn1)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/1a_test_accuracy_with_3_layer_network.png')


    # =====================Q2 Determine optimal batch size=====================

    # plot learning curves
    batch_sizes = [4,8,16,32,64]
    train_err_list = []
    test_acc_list = []
    time_taken_one_epoch_list = []

    for batch_size in batch_sizes:
        train_err, test_acc, time_taken_one_epoch = train(batch_size=batch_size, **train_test)
        train_err_list.append(train_err)
        test_acc_list.append(test_acc)
        time_taken_one_epoch_list.append(time_taken_one_epoch)

    # Plot Training Errors
    plt.figure("Train Error against Epoch with different Batch Sizes")
    plt.title("Train Error against Epoch with different Batch Sizes")
    for i in range(len(batch_sizes)):
        plt.plot(range(epochs), train_err_list[i], label = 'batch_size = {}'.format(batch_sizes[i]))
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Train Error')
        plt.legend()
        plt.grid(b=True)
        plt.savefig('figures/2a_train_error_vs_epoch_for_diff_batch_size.png')

    # Plot Test Accuracy
    plt.figure("Test Accuracy against Epoch with different Batch Sizes")
    plt.title("Test Accuracy against Epoch with different Batch Sizes")
    for i in range(len(batch_sizes)):
        plt.plot(range(epochs), test_acc_list[i], label = 'batch_size = {}'.format(batch_sizes[i]))
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
        plt.savefig('figures/2a_test_accuracy_vs_epoch_for_diff_batch_size.png')

    # Plot Time Taken for One Epoch
    plt.figure("Time Taken for One Epoch againt Batch Size")
    plt.title("Time Taken for One Epoch againt Batch Size")
    plt.plot(batch_sizes, time_taken_one_epoch_list)
    plt.xlabel('Batch Size')
    plt.ylabel('Time/ms')
    plt.grid(b=True)
    plt.savefig('figures/2b_time_taken_for_one_epoch_vs_batch_size.png')

    #plot converged test accuracy against Number of Neurons
    final_acc = [acc[-1] for acc in test_acc_list]
    plt.figure('Accuracy against Batch Size')
    plt.title('Accuracy against Batch Size')
    plt.plot(batch_sizes, final_acc)
    plt.xlabel('Batch Size')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/2c_Converged Accuracy against Batch Size.png')

    optimal_batch_size = 16

    # =====================Q3 Determine optimal number of hidden neurons=====================

    # plot learning curves
    num_hidden_neurons = [5,10,15,20,25]
    train_err_list = []
    test_acc_list = []
    time_taken_one_epoch_list = []

    for num_neurons in num_hidden_neurons:
        train_err, test_acc, time_taken_one_epoch = train(num_neurons=num_neurons, batch_size= optimal_batch_size,**train_test)
        train_err_list.append(train_err)
        test_acc_list.append(test_acc)
        time_taken_one_epoch_list.append(time_taken_one_epoch)

    # Plot Training Errors
    plt.figure("Train Error against Epoch with different Number of Neurons")
    plt.title("Train Error against Epoch with different Number of Neurons")
    for i in range(len(num_hidden_neurons)):
        plt.plot(range(epochs), train_err_list[i], label = 'num_neurons = {}'.format(num_hidden_neurons[i]))
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Train Error')
        plt.legend()
        plt.grid(b=True)
        plt.savefig('figures/3a_train_error_vs_epoch_for_diff_num_neurons.png')

    # Plot Test Accuracy
    plt.figure("Test Accuracy against Epoch with different Number of Neurons")
    plt.title("Test Accuracy against Epoch with different Number of Neurons")
    for i in range(len(num_hidden_neurons)):
        plt.plot(range(epochs), test_acc_list[i], label = 'num_neurons = {}'.format(num_hidden_neurons[i]))
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
        plt.savefig('figures/3a_test_accuracy_vs_epoch_for_diff_num_neurons.png')

    # Plot Time Taken for One Epoch
    plt.figure("Time Taken for One Epoch againt Number of Neurons")
    plt.title("Time Taken for One Epoch againt Number of Neurons")
    plt.plot(num_hidden_neurons, time_taken_one_epoch_list)
    plt.xlabel('Number of Neurons')
    plt.ylabel('Time/ms')
    plt.grid(b=True)
    plt.savefig('figures/3b_time_taken_for_one_epoch_vs_num_neurons.png')

    # plot final test accuracy against Number of Neurons
    final_acc = [acc[-1] for acc in test_acc_list]
    plt.figure('Accuracy against Number of Neurons')
    plt.title('Accuracy against Number of Neurons')
    plt.plot(num_hidden_neurons, final_acc)
    plt.xlabel('Number of Neurons')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/3c_Accuracy against Number of Neurons.png')

    optimal_num_neurons = 25

    # =====================Q4 Determine optimal decay parameter=====================

    # plot learning curves
    beta_list = [0,1e-12,1e-9,1e-6,1e-3]
    train_err_list = []
    test_acc_list = []
    time_taken_one_epoch_list = []

    for beta in beta_list:
        train_err, test_acc, time_taken_one_epoch = train(beta=beta, batch_size=optimal_batch_size,num_neurons=optimal_num_neurons, **train_test)
        train_err_list.append(train_err)
        test_acc_list.append(test_acc)
        time_taken_one_epoch_list.append(time_taken_one_epoch)

    # Plot Training Errors
    plt.figure("Train Error against Epoch with Different Decay Parameters")
    plt.title("Train Error against Epoch with Different Decay Parameters")
    for i in range(len(beta_list)):
        plt.plot(range(epochs), train_err_list[i], label = 'beta = {}'.format(beta_list[i]))
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Train Error')
        plt.legend()
        plt.grid(b=True)
        plt.savefig('figures/4a_train_error_vs_epoch_for_diff_beta.png')

    # Plot Test Accuracy against Decay Parameters
    final_acc = [acc[-1] for acc in test_acc_list]
    plt.figure('Test Accuracy against Decay Parameters')
    plt.title('Test Accuracy against Decay Parameters')
    plt.xticks(np.arange(5), [str(beta) for beta in beta_list])
    plt.plot([str(beta) for beta in beta_list], final_acc)
    plt.xlabel('Decay Parameters')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/4b_Test Accuracy against Decay Parameters.png')

    optimal_beta = 0

    # =====================Q5=====================

    train_err_ffn_list = []
    test_acc_ffn_list = []
    time_taken_one_epoch_ffn_list =[]
    train_err_ffn_list.append(train_err_ffn1)
    test_acc_ffn_list.append(test_acc_ffn1)
    time_taken_one_epoch_ffn_list.append(time_taken_one_epoch_ffn1)

    train_err_ffn2, test_acc_ffn2, time_taken_one_epoch_ffn2 = train(num_hidden_layer=2, **train_test)

    train_err_ffn_list.append(train_err_ffn2)
    test_acc_ffn_list.append(test_acc_ffn2)
    time_taken_one_epoch_ffn_list.append(time_taken_one_epoch_ffn2)

    # plot train_err against epoch
    plt.figure('Training Error: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    plt.title('Training Error: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    plt.plot(range(epochs), train_err_ffn2)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Train Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/5a_train_error_with_4_layer_network.png')

    # plot test_acc against epoch
    plt.figure('Test Accuracy: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    plt.title('Test Accuracy: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    plt.plot(range(epochs), test_acc_ffn2)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/5a_test_accuracy_with_4_layer_network.png')

    # Q5(2)
    # Plot Training Errors
    plt.figure("Train Error against Epoch with Different Number of Hidden Layers")
    plt.title("Train Error against Epoch with Different Number of Hidden Layers")
    for i in range(len(train_err_ffn_list)):
        plt.plot(range(epochs), train_err_ffn_list[i], label = 'Number of Hidden Layers = {}'.format(i+1))
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Train Error')
        plt.legend()
        plt.grid(b=True)
        plt.savefig('figures/5b_train_error_vs_epoch_for_diff_num_hidden_layers.png')

    # Plot Test Accuracy
    plt.figure("Test Accuracy against Epoch with Different Number of Hidden Layers")
    plt.title("Test Accuracy against Epoch with Different Number of Hidden Layers")
    for i in range(len(test_acc_ffn_list)):
        plt.plot(range(epochs), test_acc_ffn_list[i], label = 'Number of Hidden Layers = {}'.format(i+1))
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
        plt.savefig('figures/5b_test_accuracy_vs_epoch_for_diff_num_hidden_layers.png')

    print ("Time taken for 3_layer: %g \nTime taken for 4_layer: %g" % (time_taken_one_epoch_ffn_list[0],time_taken_one_epoch_ffn_list[1]))
#end def

if __name__ == '__main__': main()
