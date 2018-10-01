#
# Project 1, starter code part b
#

import tensorflow as tf
import numpy as np
import pylab as plt
from functools import partial
import multiprocessing as mp


NUM_FEATURES = 8

learning_rate = 0.01
epochs = 500
batch_size = 32
num_neuron = 30
seed = 10
np.random.seed(seed)


def train(trainX, trainY, testX, testY, small=False, num_hidden_layer=1, batch_size=32, num_neurons=30, learning_rate=10**(-7), beta=10**(-3), **kwargs):
    # Create the model
	x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
	y_ = tf.placeholder(tf.float32, [None, 1])

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
          tf.truncated_normal([num_neurons, 1],
                              stddev=1.0 / math.sqrt(float(num_neurons))),
          name='weights')
        b2 = tf.Variable(tf.zeros([1]), name='biases')
        y = tf.matmul(h1, w2) + b2

		error = tf.reduce_mean(tf.square(y_ - y))

		regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)

        loss = tf.reduce_mean(error + beta*regularization)

		#Create the gradient descent optimizer with the given learning rate.
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		# Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)

		mean_err = []
		

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			train_err = []
			for i in range(epochs):
				train_op.run(feed_dict={x: trainX, y_: trainY})
				err = error.eval(feed_dict={x: trainX, y_: trainY})
				train_err.append(err)

				if i % 100 == 0:
					print('iter %d: test error %g'%(i, train_err[i]))

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


def main():
	# ======================Q1======================
	#read and divide data into test and train sets
	cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
	X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
	Y_data = (np.asmatrix(Y_data)).transpose()

	idx = np.arange(X_data.shape[0])
	np.random.shuffle(idx)
	X_data, Y_data = X_data[idx], Y_data[idx]

	m = 3* X_data.shape[0] // 10
	trainX, trainY = X_data[m:], Y_data[m:]

	trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)

	# experiment with small datasets
	trainX = trainX[:1000]
	trainY = trainY[:1000]

	# ===Q2==
    no_threads = mp.cpu_count()


    probs = [0.2, 0.4, 0.6, 1.0]

	p = mp.Pool(processes = no_threads)

    acc = p.map(train, probs)







# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_err)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train Error')
plt.show()
