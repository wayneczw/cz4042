#
# Project 1, starter code part b
#

# import matplotlib
# matplotlib.use('Agg')
import math
import matplotlib.pylab as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time

TEST_SIZE = 0.3
NUM_FEATURES = 8
seed = 10
np.random.seed(seed)

if not os.path.exists('figures'):
    os.makedirs('figures')
if not os.path.exists(os.path.join('figures', '1b')):
    os.makedirs(os.path.join('figures', '1b'))

class CVRegressor():
    def __init__(
        self,
        features_dim=None, output_dim=None, drop_out=False,
        keep_prob=0.9, num_folds=5, num_hidden_layers=1,
        hidden_layer_dict={1: 30},
        batch_size=32, learning_rate=1e-7,
        l2_beta=1e-3, epochs=500, set_session=True, tf_config=None,
        **kwargs
    ):

        self.features_dim = features_dim
        self.output_dim = output_dim
        self.drop_out = drop_out
        if self.drop_out:
            self._keep_prob = tf.placeholder(tf.float32)
            self.keep_prob = keep_prob
        self.num_folds = num_folds
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_dict = hidden_layer_dict
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_beta = l2_beta
        self.epochs = epochs

        self._build_model()
    #end def


    def _build_layer(self, X, input_dim, output_dim, hidden=False):
        W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1.0/math.sqrt(float(input_dim)), seed=10), name='weights')
        B = tf.Variable(tf.zeros([output_dim]), name='biases')
        if hidden:
            U = tf.nn.relu(tf.matmul(X, W) + B)
            if self.drop_out:
                U = tf.nn.dropout(U, self._keep_prob)
                return W, B, U
        else:
            U = tf.matmul(X, W) + B

        return W, B, U
    #end def


    def _build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.features_dim])
        self.y_ = tf.placeholder(tf.float32, [None, self.output_dim])

        if self.num_hidden_layers == 1:
            self.W, self.B, self.H = self._build_layer(self.x, self.features_dim, self.hidden_layer_dict[1], hidden=True)
            self.V, self.C, self.U = self._build_layer(self.H, self.hidden_layer_dict[1], self.output_dim)
            self.regularization = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.V)
        elif self.num_hidden_layers == 2:
            self.W, self.B, self.H = self._build_layer(self.x, self.features_dim, self.hidden_layer_dict[1], hidden=True)
            self.G, self.J, self.R = self._build_layer(self.H, self.hidden_layer_dict[1], self.hidden_layer_dict[2], hidden=True)
            self.V, self.C, self.U = self._build_layer(self.R, self.hidden_layer_dict[2], self.output_dim)
            self.regularization = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.G) + tf.nn.l2_loss(self.V)
        elif self.num_hidden_layers == 3:
            self.W, self.B, self.H = self._build_layer(self.x, self.features_dim, self.hidden_layer_dict[1], hidden=True )
            self.G, self.J, self.R = self._build_layer(self.H, self.hidden_layer_dict[1], self.hidden_layer_dict[2], hidden=True)
            self.S, self.A, self.N = self._build_layer(self.R, self.hidden_layer_dict[2], self.hidden_layer_dict[3], hidden=True)
            self.V, self.C, self.U = self._build_layer(self.N, self.hidden_layer_dict[3], self.output_dim)
            self.regularization = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.G) + tf.nn.l2_loss(self.S) + tf.nn.l2_loss(self.V)

        self.error = tf.reduce_mean(tf.reduce_sum(tf.square(self.U - self.y_), axis = 1))
        self.loss = tf.reduce_mean(self.error + self.l2_beta*self.regularization)

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
    #end def

    def trainCV(self, trainX, trainY, small=False):
        np.random.seed(10)
        if small:
            trainX = trainX[:300]
            trainY = trainY[:300]

        _cv_err = []
        N = int(len(trainX) / self.num_folds)  # number of instances in each fold
        idx = np.arange(N)
        time_to_update = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            for fold in range(self.num_folds):
                start, end = int(fold * N), int((fold + 1) * N)
                X_val, Y_val = trainX[start:end], trainY[start:end]
                X_train = np.append(trainX[:start], trainX[end:], axis=0)
                Y_train = np.append(trainY[:start], trainY[end:], axis=0)

                _N = len(X_train)
                # scale at each fold
                x_mean = np.mean(X_train, axis=0)
                x_std = np.std(X_train, axis=0)
                X_train = scale(X_train, x_mean, x_std)
                X_val = scale(X_val, x_mean, x_std)

                for i in range(self.epochs):
                    np.random.shuffle(idx)
                    X_train = X_train[idx]
                    Y_train = Y_train[idx]

                    t = time.time()
                    for _start, _end in zip(range(0, _N, self.batch_size), range(self.batch_size, _N, self.batch_size)):
                        if self.drop_out:
                            self.train_op.run(feed_dict={self.x: X_train[_start:_end], self.y_: Y_train[_start:_end], self._keep_prob: self.keep_prob})
                        else:
                            self.train_op.run(feed_dict={self.x: X_train[_start:_end], self.y_: Y_train[_start:_end]})
                    time_to_update += (time.time() - t)

                    if i % 100 == 0:
                        print('fold %g: iter %d'%(fold, i))
                        print('----------------------')

                #end for
                if self.drop_out:
                    _cv_err.append(self.loss.eval(feed_dict={self.x: X_val, self.y_: Y_val, self._keep_prob: self.keep_prob}))
                else:
                    _cv_err.append(self.loss.eval(feed_dict={self.x: X_val, self.y_: Y_val}))
            #end for
            self.saver.save(sess, ".ckpt/1bmodel.ckpt")
        #end with
        self.cv_err = np.mean(np.array(_cv_err), axis=0)
        self.time_taken_one_epoch = (time_to_update/(self.epochs * self.num_folds)) * 1000
        return self
    #end def


    def train(self, trainX, trainY, testX, testY, small=False):
        np.random.seed(10)
        if small:
            trainX = trainX[:300]
            trainY = trainY[:300]

        N = len(trainX)
        idx = np.arange(N)
        time_to_update = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            # scale
            x_mean = np.mean(trainX, axis=0)
            x_std = np.std(trainX, axis=0)

            X_train = scale(trainX, x_mean, x_std)
            X_test = testX
            Y_train = trainY
            Y_test = testY

            self.train_err = []
            self.test_err = []
            for i in range(self.epochs):
                np.random.shuffle(idx)
                X_train = X_train[idx]
                Y_train = Y_train[idx]

                t = time.time()
                for _start, _end in zip(range(0, N, self.batch_size), range(self.batch_size, N, self.batch_size)):
                    if self.drop_out:
                        self.train_op.run(feed_dict={self.x: X_train[_start:_end], self.y_: Y_train[_start:_end], self._keep_prob: self.keep_prob})
                    else:
                        self.train_op.run(feed_dict={self.x: X_train[_start:_end], self.y_: Y_train[_start:_end]})

                time_to_update += (time.time() - t)

                if self.drop_out:
                    self.train_err.append(self.loss.eval(feed_dict={self.x: X_train, self.y_: Y_train, self._keep_prob: self.keep_prob}))
                    self.test_err.append(self.error.eval(feed_dict={self.x: X_test, self.y_: Y_test, self._keep_prob: 1.0}))
                else:
                    self.train_err.append(self.loss.eval(feed_dict={self.x: X_train, self.y_: Y_train}))
                    self.test_err.append(self.error.eval(feed_dict={self.x: X_test, self.y_: Y_test}))

                if i % 100 == 0:
                    print('iter %d: validation error %g'%(i, self.train_err[i]))
                    print('----------------------')

            #end for
            self.saver.save(sess, ".ckpt/1bmodel.ckpt")
        #end with

        self.time_taken_one_epoch = (time_to_update/self.epochs) * 1000
        return self
    #end def

    def test(self, X_test, Y_test):
        with tf.Session() as sess:
            self.saver.restore(sess, ".ckpt/1bmodel.ckpt")
            if self.drop_out:
                feed_dict = {self.x: X_test, self.y_: Y_test, self._keep_prob: 1.0}
            else:
                feed_dict = {self.x: X_test, self.y_: Y_test}

            test_error = self.error.eval(feed_dict)
        #end with

        return test_error
    #end def


    def predict(self, X):
        with tf.Session() as sess:
            self.saver.restore(sess, ".ckpt/1bmodel.ckpt")

            if self.drop_out:
                feed_dict = {self.x: X, self._keep_prob: 1.0}
            else:
                feed_dict = {self.x: X}               
            prediction = self.U.eval(feed_dict)
        #end with

        return prediction
    #end def
#end class


# scale data
def scale(X, X_mean, X_std):
    return (X - X_mean) / X_std
#end def


def _read_data(file_name):
    _input = np.loadtxt(file_name, delimiter=',')
    _input_train, _input_test = train_test_split(_input, test_size=TEST_SIZE)

    # split data
    X_train, Y_train = _input_train[:, :8], _input_train[:, -1].astype(float)
    X_test, Y_test = _input_test[:, :8], _input_test[:, -1].astype(float)

    # scale
    x_mean = np.mean(X_train, axis=0)
    x_std = np.std(X_train, axis=0)
    X_test = scale(X_test, x_mean, x_std)

    return X_train, Y_train.reshape(len(Y_train), 1), X_test, Y_test.reshape(len(Y_test), 1)
#end def


def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b, 'y = {:.2f} + {:.2f}x'.format(a, b)
#end def


def correct_fit(X, Y):
    return a, b
#end def

def main():
    # read data
    X_train, Y_train, X_test, Y_test = _read_data('./data/cal_housing.data')

    ########### Q1 3-layer Feedforward Network ############
    regressor = CVRegressor(features_dim=NUM_FEATURES, output_dim=1,
                            hidden_layer_dict={1: 30})
    regressor = regressor.train(trainX=X_train, trainY=Y_train,
                                testX=X_test, testY=Y_test,
                                small=False)

    plt.figure('Validation Error against Epochs')
    plt.title('Validation Error against Epochs')
    plt.grid(b=True)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Error')
    plt.plot(range(1, 501), regressor.train_err)
    plt.savefig('figures/1b/1a_validation_error_with_epochs.png')

    idx = np.random.choice(len(X_test), 50, replace=True)
    Y_predict, Y_true = regressor.predict(X_test[idx]), Y_test[idx]
    Y_predict = [l[0] for l in Y_predict]
    Y_true = [l[0] for l in Y_true]
    a, b, eqn = best_fit(Y_true, Y_predict)
    # plot points and fit line
    plt.figure('Predicted Value against True Value')
    plt.title('Predicted Value against True Value')
    plt.scatter(Y_true, Y_predict)
    yfit = [a + b * xi for xi in Y_true]
    fit = Y_true
    plt.plot(Y_true, yfit, label = 'Best fit line: {}'.format(eqn))
    plt.plot(Y_true, Y_true,zu label = 'True value line: {}'.format('y = x'))
    plt.legend()
    plt.grid(b=True)
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.savefig('figures/1b/1b_Predicted_against_True.png')

    ########### Q2 Optimal Learning Rate ############
    learning_rate_list = [10**(-10), 10**(-9), 0.5 * 10**(-8), 10**(-7), 0.5 * 10**(-6)]

    CV_list = []
    time_taken_one_epoch_list = []

    for learning_rate in learning_rate_list:
        regressor = CVRegressor(features_dim=NUM_FEATURES, output_dim=1,
                                hidden_layer_dict={1: 30}, learning_rate=learning_rate)
        regressor = regressor.trainCV(trainX=X_train, trainY=Y_train, small=False)
        CV_list.append(regressor.cv_err)
        time_taken_one_epoch_list.append(regressor.time_taken_one_epoch)
    #end for

    plt.figure('CV Error against Learning Rate')
    plt.title('CV Error against Learning Rate')
    plt.grid(b=True)
    plt.xlabel('Learning Rate')
    plt.ylabel('CV Error')
    plt.xticks(np.arange(5), [str(l) for l in learning_rate_list])
    plt.plot([str(l) for l in learning_rate_list], CV_list)
    plt.savefig('figures/1b/2a_CV_error_against_learning_rate.png')

    # Plot Time Taken for One Epoch
    plt.figure("Time Taken for One Epoch against Learning Rate")
    plt.title("Time Taken for One Epoch against Learning Rate")
    plt.xlabel('Learning Rate')
    plt.ylabel('Time/ms')
    plt.xticks(np.arange(5), [str(l) for l in learning_rate_list])
    plt.plot([str(l) for l in learning_rate_list], time_taken_one_epoch_list)
    plt.grid(b=True)
    plt.savefig('figures/1b/2b_time_taken_for_one_epoch_vs_learning_rate.png')


    print("="*50)
    print("="*25 + "Results" + "="*25 )
    print("="*50)
    for i in range(len(learning_rate_list)):
        print('Learning Rate: {}'.format(learning_rate_list[i]))
        print('CV error: {}'.format(CV_list[i]))
        print('Time per epoch: {}ms'.format(time_taken_one_epoch_list[i]))
        print('-'*50)
    #end for

    # ==================================================
    # =========================Results=========================
    # ==================================================
    # Learning Rate: 1e-10
    # CV error: 30830280704.0
    # Time per epoch: 102.26419515609741ms
    # --------------------------------------------------
    # Learning Rate: 1e-09
    # CV error: 4589405696.0
    # Time per epoch: 102.4727349281311ms
    # --------------------------------------------------
    # Learning Rate: 5e-09
    # CV error: 4246615296.0
    # Time per epoch: 99.82456617355346ms
    # --------------------------------------------------
    # Learning Rate: 1e-07
    # CV error: 3947042816.0
    # Time per epoch: 99.8965838432312ms
    # --------------------------------------------------
    # Learning Rate: 5e-07
    # CV error: 4366555136.0
    # Time per epoch: 100.57755069732666ms
    # --------------------------------------------------

    optimal_learning_rate = 10**(-7)

    regressor = CVRegressor(features_dim=NUM_FEATURES, output_dim=1,
                            hidden_layer_dict={1: 30}, learning_rate=optimal_learning_rate)
    regressor = regressor.train(trainX=X_train, trainY=Y_train, testX=X_test, testY=Y_test,
                                small=False)

    plt.figure('{}: Test Error against Epochs'.format(optimal_learning_rate))
    plt.title('{}: Test Error against Epochs'.format(optimal_learning_rate))
    plt.grid(b=True)
    plt.xlabel('Epochs')
    plt.ylabel('Test Error')
    plt.plot(range(1, 501), regressor.test_err)
    plt.savefig('figures/1b/2b_test_error_with_epochs.png')


    ############ Q3 3-layer Feedforward Network ############
    num_neurons_list = [20, 40, 60, 80, 100]
    CV_list = []
    time_taken_one_epoch_list = []

    for num_neurons in num_neurons_list:
        regressor = CVRegressor(features_dim=NUM_FEATURES, output_dim=1,
                                hidden_layer_dict={1: num_neurons}, learning_rate=optimal_learning_rate)
        regressor = regressor.trainCV(trainX=X_train, trainY=Y_train, small=False)
        CV_list.append(regressor.cv_err)
        time_taken_one_epoch_list.append(regressor.time_taken_one_epoch)
    #end for

    plt.figure('CV Error against Number of Neurons')
    plt.title('CV Error against Number of Neurons')
    plt.grid(b=True)
    plt.xlabel('Number of Neurons')
    plt.ylabel('CV Error')
    plt.xticks(np.arange(5), [str(l) for l in num_neurons_list])
    plt.plot([str(l) for l in num_neurons_list], CV_list)
    plt.savefig('figures/1b/3a_CV_error_against_num_neurons.png')

    # Plot Time Taken for One Epoch
    plt.figure("Time Taken for One Epoch against Number of Neurons")
    plt.title("Time Taken for One Epoch against Number of Neurons")
    plt.xlabel('Number of Neurons')
    plt.ylabel('Time/ms')
    plt.plot(num_neurons_list, time_taken_one_epoch_list)
    plt.grid(b=True)
    plt.savefig('figures/1b/3c_time_taken_for_one_epoch_vs_num_neurons.png')

    print("="*50)
    print("="*25 + "Results" + "="*25 )
    print("="*50)
    for i in range(len(num_neurons_list)):
        print('Number of Neurons: {}'.format(num_neurons_list[i]))
        print('CV error: {}'.format(CV_list[i]))
        print('Time per epoch: {}ms'.format(time_taken_one_epoch_list[i]))
        print('-'*50)
    #end for

    # ==================================================
    # =========================Results=========================
    # ==================================================
    # Number of Neurons: 20
    # CV error: 4086170368.0
    # Time per epoch: 102.01765747070313ms
    # --------------------------------------------------
    # Number of Neurons: 40
    # CV error: 4057242112.0
    # Time per epoch: 98.3443314552307ms
    # --------------------------------------------------
    # Number of Neurons: 60
    # CV error: 4011250432.0
    # Time per epoch: 97.78651485443116ms
    # --------------------------------------------------
    # Number of Neurons: 80
    # CV error: 4020797440.0
    # Time per epoch: 100.36348514556884ms
    # --------------------------------------------------
    # Number of Neurons: 100
    # CV error: 3934771712.0
    # Time per epoch: 106.26908721923827ms
    # --------------------------------------------------
    optimal_num_neurons = 100

    regressor = CVRegressor(features_dim=NUM_FEATURES, output_dim=1,
                            hidden_layer_dict={1: optimal_num_neurons}, learning_rate=optimal_learning_rate)
    regressor = regressor.train(trainX=X_train, trainY=Y_train, testX=X_test, testY=Y_test,
                                small=False)

    plt.figure('{}: Test Error against Epochs'.format(optimal_num_neurons))
    plt.title('{}: Test Error against Epochs'.format(optimal_num_neurons))
    plt.grid(b=True)
    plt.xlabel('Epochs')
    plt.ylabel('Test Error')
    plt.plot(range(1, 501), regressor.test_err)
    plt.savefig('figures/1b/3bb_test_error_with_epochs.png')


    ############ Q4 3-layer Feedforward Network ############
    state_list = ['3L w/o dp', '3L w/ dp', '4L w/o dp', '4L w/ dp', '5L w/o dp', '5L w/ dp']
    test_error_list = []
    time_taken_one_epoch_list = []

    #### 3-layer without dropout ####
    regressor = CVRegressor(features_dim=NUM_FEATURES, output_dim=1,
                            hidden_layer_dict={1: optimal_num_neurons},
                            num_hidden_layers=1, learning_rate=10**(-9))
    regressor = regressor.train(trainX=X_train, trainY=Y_train, testX=X_test, testY=Y_test,
                                small=False)
    test_error_list.append(regressor.test(X_test, Y_test))
    time_taken_one_epoch_list.append(regressor.time_taken_one_epoch)

    #### 3-layer with dropout ####
    regressor = CVRegressor(features_dim=NUM_FEATURES, output_dim=1,
                            hidden_layer_dict={1: optimal_num_neurons},
                            num_hidden_layers=1, drop_out=True, keep_prob=0.9,
                            learning_rate=10**(-9))
    regressor = regressor.train(trainX=X_train, trainY=Y_train, testX=X_test, testY=Y_test,
                                small=False)
    test_error_list.append(regressor.test(X_test, Y_test))
    time_taken_one_epoch_list.append(regressor.time_taken_one_epoch)

    #### 4-layer without dropout ####
    regressor = CVRegressor(features_dim=NUM_FEATURES, output_dim=1,
                            hidden_layer_dict={1: optimal_num_neurons, 2: 20},
                            num_hidden_layers=2, learning_rate=10**(-9))
    regressor = regressor.train(trainX=X_train, trainY=Y_train, testX=X_test, testY=Y_test,
                                small=False)
    test_error_list.append(regressor.test(X_test, Y_test))
    time_taken_one_epoch_list.append(regressor.time_taken_one_epoch)

    #### 4-layer with dropout ####
    regressor = CVRegressor(features_dim=NUM_FEATURES, output_dim=1,
                            hidden_layer_dict={1: optimal_num_neurons, 2: 20},
                            num_hidden_layers=2, drop_out=True, keep_prob=0.9,
                            learning_rate=10**(-9))
    regressor = regressor.train(trainX=X_train, trainY=Y_train, testX=X_test, testY=Y_test,
                                small=False)
    test_error_list.append(regressor.test(X_test, Y_test))
    time_taken_one_epoch_list.append(regressor.time_taken_one_epoch)

    #### 5-layer without dropout ####
    regressor = CVRegressor(features_dim=NUM_FEATURES, output_dim=1,
                            hidden_layer_dict={1: optimal_num_neurons, 2: 20, 3: 20},
                            num_hidden_layers=3, learning_rate=10**(-9))
    regressor = regressor.train(trainX=X_train, trainY=Y_train, testX=X_test, testY=Y_test,
                                small=False)
    test_error_list.append(regressor.test(X_test, Y_test))
    time_taken_one_epoch_list.append(regressor.time_taken_one_epoch)

    #### 5-layer with dropout ####
    regressor = CVRegressor(features_dim=NUM_FEATURES, output_dim=1,
                            hidden_layer_dict={1: optimal_num_neurons, 2: 20, 3: 20},
                            num_hidden_layers=3, drop_out=True, keep_prob=0.9,
                            learning_rate=10**(-9))
    regressor = regressor.train(trainX=X_train, trainY=Y_train, testX=X_test, testY=Y_test,
                                small=False)
    test_error_list.append(regressor.test(X_test, Y_test))
    time_taken_one_epoch_list.append(regressor.time_taken_one_epoch)

    for i in range(len(state_list)):
        print("state list: {}, test_error: {}".format(state_list[i],test_error_list[i]))
    # state list: 3L w/o dp, test_error: 4523873280.0
    # state list: 3L w/ dp, test_error: 4526773248.0
    # state list: 4L w/o dp, test_error: 3021156864.0
    # state list: 4L w/ dp, test_error: 3602084864.0
    # state list: 5L w/o dp, test_error: 2649017856.0
    # state list: 5L w/ dp, test_error: 3052726528.0


    # Plot Time Taken for One Epoch
    plt.figure("Time Taken for One Epoch againt state")
    plt.title("Time Taken for One Epoch againt state")
    plt.grid(b=True)
    plt.ylabel('Time/ms')
    plt.xticks(np.arange(6), state_list)
    plt.plot(state_list, time_taken_one_epoch_list)
    plt.savefig('figures/1b/4_Time.png')

    plt.figure('Test Error')
    plt.title('Test Error')
    plt.grid(b=True)
    plt.ylabel('Test Error')
    plt.xticks(np.arange(6), state_list)
    plt.plot(state_list, test_error_list)
    plt.savefig('figures/1b/4_Test_error.png')

    print("="*50)
    print("="*25 + "Results" + "="*25 )
    print("="*50)
    for i in range(len(state_list)):
        print('State: {}'.format(state_list[i]))
        print('Time per epoch: {}ms'.format(time_taken_one_epoch_list[i]))
        print('Test Error: {}'.format(test_error_list[i]))
        print('-'*50)
    #end for
    # ==================================================
    # =========================Results=========================
    # ==================================================
    # State: 3L w/o dp
    # Time per epoch: 137.84410953521729ms
    # Test Error: 4523873280.0
    # --------------------------------------------------
    # State: 3L w/ dp
    # Time per epoch: 171.21195077896118ms
    # Test Error: 4526773248.0
    # --------------------------------------------------
    # State: 4L w/o dp
    # Time per epoch: 162.18966579437256ms
    # Test Error: 3021156864.0
    # --------------------------------------------------
    # State: 4L w/ dp
    # Time per epoch: 206.2958059310913ms
    # Test Error: 3602084864.0
    # --------------------------------------------------
    # State: 5L w/o dp
    # Time per epoch: 183.60487985610962ms
    # Test Error: 2649017856.0
    # --------------------------------------------------
    # State: 5L w/ dp
    # Time per epoch: 239.46327114105225ms
    # Test Error: 3052726528.0
    # --------------------------------------------------


#end def


if __name__ == '__main__': main()
