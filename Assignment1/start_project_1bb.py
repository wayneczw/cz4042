#
# Project 1, starter code part b
#

# import matplotlib
# matplotlib.use('Agg')
import math
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import time

TEST_SIZE = 0.3
NUM_FEATURES = 8
seed = 10
np.random.seed(seed)


class CVRegressor():
    def __init__(
        self,
        features_dim=None, output_dim=None, drop_out=False,
        keep_prob=0.9, num_folds=5, num_hidden_layers=1,
        hidden_layer_dict={1: 30},
        batch_size=32, learning_rate=10**(-7),
        l2_beta=10**(-3), epochs=500, set_session=True, tf_config=None,
        **kwargs
    ):

        self.features_dim = features_dim
        self.output_dim = output_dim
        self.drop_out = drop_out
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
                U = tf.nn.dropout(U, self.keep_prob)
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

    def train(self, trainX, trainY, testX=[], testY=[], small=False):
        np.random.seed(10)
        if small:
            trainX = trainX[:300]
            trainY = trainY[:300]

        _cv_err = []
        test_err = []
        val_err = []
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

                # scale at each fold
                x_mean = np.mean(X_train, axis=0)
                x_std = np.std(X_train, axis=0)
                X_train = scale(X_train, x_mean, x_std)
                X_val = scale(X_val, x_mean, x_std)

                _val_err = []
                _test_err = []
                for i in range(self.epochs):
                    np.random.shuffle(idx)
                    X_train = X_train[idx]
                    Y_train = Y_train[idx]

                    t = time.time()
                    for _start, _end in zip(range(0, N, self.batch_size), range(self.batch_size, N, self.batch_size)):
                        self.train_op.run(feed_dict={self.x: X_train[_start:_end], self.y_: Y_train[_start:_end]})
                    time_to_update += (time.time() - t)

                    _val_err.append(self.loss.eval(feed_dict={self.x: X_val, self.y_: Y_val}))

                    if len(testX)>1:
                        # prediction = self.U.eval({self.x: testX})
                        # _test_err.append(tf.losses.mean_squared_error(testY, prediction))
                        _test_err.append(self.error.eval(feed_dict={self.x: testX, self.y_: testY}))
                    if i % 100 == 0:
                        print('fold %g: iter %d: validation error %g'%(fold, i, _val_err[i]))
                        print('----------------------')

                #end for
                val_err.append(_val_err)
                _cv_err.append(self.loss.eval(feed_dict={self.x: X_val, self.y_: Y_val}))
                if len(testX)>1: test_err.append(_test_err)
            #end for
            self.saver.save(sess, ".ckpt/1bmodel.ckpt")
        #end with
        self.val_err = np.mean(np.array(val_err), axis=0)
        self.cv_err = np.mean(np.array(_cv_err), axis=0)
        if len(testX)>1:
            test_err = np.array(test_err)
            self.test_err = np.mean(test_err, axis=0)
        #end if
        self.time_taken_one_epoch = (time_to_update/(self.epochs * self.num_folds)) * 1000
        return self
   #end def


    def test(self, X_test, Y_test):
        with tf.Session() as sess:
            self.saver.restore(sess, ".ckpt/1bmodel.ckpt")
            test_error = self.error.eval(feed_dict={self.x: X_test, self.y_: Y_test})
        #end with

        return test_error
    #end def


    def predict(self, X):
        with tf.Session() as sess:
            self.saver.restore(sess, ".ckpt/1bmodel.ckpt")

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

    # scaling test data only
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

    return a, b
#end def


def main():
    # read data
    X_train, Y_train, X_test, Y_test = _read_data('./data/cal_housing.data')

    ########### Q1 3-layer Feedforward Network ############
    regressor = CVRegressor(features_dim=NUM_FEATURES, output_dim=1,
                            hidden_layer_dict={1: 30})
    regressor = regressor.train(trainX=X_train, trainY=Y_train,
                                small=False)

    plt.figure('Validation Error against Epochs')
    plt.title('Validation Error against Epochs')
    plt.grid(b=True)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Error')
    plt.plot(range(1, 501), regressor.val_err)
    plt.savefig('figures/1b/1a_validation_error_with_epochs.png')

    idx = np.random.choice(len(X_test), 50, replace=True)
    Y_predict, Y_true = regressor.predict(X_test[idx]), Y_test[idx]
    Y_predict = [l[0] for l in Y_predict]
    Y_true = [l[0] for l in Y_true]
    a, b = best_fit(Y_true, Y_predict)
    # plot points and fit line
    plt.figure('Predicted Value against True Value')
    plt.title('Predicted Value against True Value')
    plt.scatter(Y_true, Y_predict)
    yfit = [a + b * xi for xi in Y_true]
    plt.plot(Y_true, yfit)
    plt.grid(b=True)
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.savefig('figures/1b/1b_Predicted_against_True.png')

    ########### Q2 Optimal Learning Rate ############
    learning_rate_list = [10**(-10), 10**(-9), 0.5 * 10**(-8), 10**(-7), 0.5 * 10**(-6)]

    CV_list = []
    test_error_list = []
    time_taken_one_epoch_list = []

    for learning_rate in learning_rate_list:
        regressor = CVRegressor(features_dim=NUM_FEATURES, output_dim=1,
                                hidden_layer_dict={1: 30}, learning_rate=learning_rate)
        regressor = regressor.train(trainX=X_train, trainY=Y_train, testX=X_test, testY=Y_test,
                                small=False)
        CV_list.append(regressor.cv_err)
        test_error_list.append(regressor.test_err)
        time_taken_one_epoch_list.append(regressor.time_taken_one_epoch)
    #end for

    plt.figure('CV Error against Learning Rate')
    plt.title('CV Error against Learning Rate')
    plt.grid(b=True)
    plt.xlabel('Learning Rate')
    plt.ylabel('CV Error')
    plt.xticks(np.arange(5), [str(l) for l in learning_rate_list])
    plt.plot([str(l) for l in learning_rate_list], CV_list)
    plt.savefig('figures/1b/2aCV_error_against_learning_rate.png')

    plt.figure('Test Error against Epochs for each Learning Rate')
    plt.title('Test Error against Epochs')
    plt.grid(b=True)
    plt.xlabel('Epochs')
    plt.ylabel('Test Error')
    for i in range(len(learning_rate_list)):
        plt.plot(range(1, 501), test_error_list[i], label='Learning Rate {}'.format(learning_rate_list[i]))
        plt.legend()
    #end for
    plt.savefig('figures/1b/2bTest_error_against_epochs.png')

    # Plot Time Taken for One Epoch
    plt.figure("Time Taken for One Epoch against Learning Rate")
    plt.title("Time Taken for One Epoch against Learning Rate")
    plt.xlabel('Learning Rate')
    plt.ylabel('Time/ms')
    plt.xticks(np.arange(5), [str(l) for l in learning_rate_list])
    plt.plot([str(l) for l in learning_rate_list], time_taken_one_epoch_list)
    plt.grid(b=True)
    plt.savefig('figures/1b/2b_time_taken_for_one_epoch_vs_learning_rate.png')

    # plot final test error against learning rate
    final_err = [err[-1] for err in test_error_list]
    plt.figure('Converged Error against Learning Rate')
    plt.title('Converged Error against Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Test Error')
    plt.xticks(np.arange(5), [str(l) for l in learning_rate_list])
    plt.plot([str(l) for l in learning_rate_list], final_err)
    plt.grid(b=True)
    plt.savefig('figures/1a/2b_error_against_learning_rate.png')

    print("="*50)
    print("="*25 + "Results" + "="*25 )
    print("="*50)
    for i in range(len(learning_rate_list)):
        print('Learning Rate: {}'.format(learning_rate_list[i]))
        print('Time per epoch: {}ms'.format(time_taken_one_epoch_list[i]))
        print('Convergence Test Error: {}'.format(final_err[i]))
        print('-'*50)
    #end for

    # ==================================================
    # =========================Results=========================
    # ==================================================
    # Learning Rate: 1e-10
    # Time per epoch: 52.906027984619136ms
    # Convergence Test Error: 31665864704.0
    # --------------------------------------------------
    # Learning Rate: 1e-09
    # Time per epoch: 50.655290031433104ms
    # Convergence Test Error: 4719691264.0
    # --------------------------------------------------
    # Learning Rate: 5e-09
    # Time per epoch: 50.2107159614563ms
    # Convergence Test Error: 4405347328.0
    # --------------------------------------------------
    # Learning Rate: 1e-07
    # Time per epoch: 50.12419328689575ms
    # Convergence Test Error: 4175657728.0
    # --------------------------------------------------
    # Learning Rate: 5e-07
    # Time per epoch: 49.85496969223022ms
    # Convergence Test Error: 4366656512.0
    # --------------------------------------------------

    optimal_learning_rate = 1 * 10**(-7)

    ############ Q3 3-layer Feedforward Network ############
    num_neurons_list = [20, 40, 60, 80, 100]
    CV_list = []
    test_error_list = []
    time_taken_one_epoch_list = []

    for num_neurons in num_neurons_list:
        regressor = CVRegressor(features_dim=NUM_FEATURES, output_dim=1,
                                hidden_layer_dict={1: num_neurons}, learning_rate=optimal_learning_rate)
        regressor = regressor.train(trainX=X_train, trainY=Y_train, testX=X_test, testY=Y_test,
                                small=False)
        CV_list.append(regressor.cv_err)
        test_error_list.append(regressor.test_err)
        time_taken_one_epoch_list.append(regressor.time_taken_one_epoch)
    #end for

    plt.figure('CV Error against Number of Neurons')
    plt.title('CV Error against Number of Neurons')
    plt.grid(b=True)
    plt.xlabel('Number of Neurons')
    plt.ylabel('CV Error')
    plt.xticks(np.arange(5), [str(l) for l in num_neurons_list])
    plt.plot([str(l) for l in num_neurons_list], CV_list)
    plt.savefig('figures/1b/3aCV_error_against_num_neurons.png')

    plt.figure('Num Neurons - Test Error against Epochs')
    plt.title('Test Error against Epochs')
    plt.grid(b=True)
    plt.xlabel('Epochs')
    plt.ylabel('Test Error')
    for i in range(len(num_neurons_list)):
        plt.plot(range(1, 501), test_error_list[i], label='Number of Neurons {}'.format(num_neurons_list[i]))
        plt.legend()
    #end for
    plt.savefig('figures/1b/3bTest_error_against_epochs.png')

    # Plot Time Taken for One Epoch
    plt.figure("Time Taken for One Epoch against Number of Neurons")
    plt.title("Time Taken for One Epoch against Number of Neurons")
    plt.xlabel('Number of Neurons')
    plt.ylabel('Time/ms')
    plt.plot(num_neurons_list, time_taken_one_epoch_list)
    plt.grid(b=True)
    plt.savefig('figures/1b/3c_time_taken_for_one_epoch_vs_num_neurons.png')

    # plot final test error against num neurons
    final_err = [err[-1] for err in test_error_list]
    plt.figure('Converged Error against Number of Neurons')
    plt.title('Converged Error against Number of Neurons')
    plt.plot(num_neurons_list, final_err)
    plt.xlabel('Number of Neurons')
    plt.ylabel('Test Error')
    plt.grid(b=True)
    plt.savefig('figures/1a/3c_error_against_num_neurons.png')

    print("="*50)
    print("="*25 + "Results" + "="*25 )
    print("="*50)
    for i in range(len(num_neurons_list)):
        print('Number of Neurons: {}'.format(num_neurons_list[i]))
        print('Time per epoch: {}ms'.format(time_taken_one_epoch_list[i]))
        print('Convergence Test Error: {}'.format(final_err[i]))
        print('-'*50)
    #end for
    # ==================================================
    # =========================Results=========================
    # ==================================================
    # Number of Neurons: 20
    # Time per epoch: 26.007418441772458ms
    # Convergence Test Error: 4205488128.0
    # --------------------------------------------------
    # Number of Neurons: 40
    # Time per epoch: 25.636677455902102ms
    # Convergence Test Error: 4129115648.0
    # --------------------------------------------------
    # Number of Neurons: 60
    # Time per epoch: 25.988332271575928ms
    # Convergence Test Error: 4167387648.0
    # --------------------------------------------------
    # Number of Neurons: 80
    # Time per epoch: 26.08174295425415ms
    # Convergence Test Error: 4216478976.0
    # --------------------------------------------------
    # Number of Neurons: 100
    # Time per epoch: 27.182005214691163ms
    # Convergence Test Error: 4066954496.0
    # --------------------------------------------------
    optimal_num_neurons = 40

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

    plt.figure('Test Error')
    plt.title('Test Error')
    plt.grid(b=True)
    plt.ylabel('Test Error')
    plt.xticks(np.arange(6), state_list)
    plt.plot(state_list, test_error_list)
    plt.savefig('figures/1b/4Test_error.png')

    print("="*50)
    print("="*25 + "Results" + "="*25 )
    print("="*50)
    for i in range(len(state_list)):
        print('State: {}'.format(state_list[i]))
        print('Time per epoch: {}ms'.format(time_taken_one_epoch_list[i]))
        print('Convergence Test Error: {}'.format(test_error_list[i]))
        print('-'*50)
    #end for
    # ==================================================
    # =========================Results=========================
    # ==================================================
    # State: 3L w/o dp
    # Time per epoch: 47.23421764373779ms
    # Convergence Test Error: 4540545536.0
    # --------------------------------------------------
    # State: 3L w/ dp
    # Time per epoch: 53.36572380065918ms
    # Convergence Test Error: 4972498944.0
    # --------------------------------------------------
    # State: 4L w/o dp
    # Time per epoch: 56.572142887115476ms
    # Convergence Test Error: 3475232256.0
    # --------------------------------------------------
    # State: 4L w/ dp
    # Time per epoch: 64.23646202087401ms
    # Convergence Test Error: 4715468800.0
    # --------------------------------------------------
    # State: 5L w/o dp
    # Time per epoch: 64.29392166137696ms
    # Convergence Test Error: 3774078976.0
    # --------------------------------------------------
    # State: 5L w/ dp
    # Time per epoch: 76.36065416336059ms
    # Convergence Test Error: 4738519552.0
    # --------------------------------------------------

#end def


if __name__ == '__main__': main()
