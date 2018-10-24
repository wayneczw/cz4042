#
# Project 1, starter code part a
#
from keras import backend as K

import logging
import math
import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import time
import os

if not os.path.exists('figures'):
    os.makedirs('figures')
if not os.path.exists(os.path.join('figures', '1a')):
    os.makedirs(os.path.join('figures', '1a'))

logger = logging.getLogger(__name__)
NUM_FEATURES = 36
NUM_CLASSES = 6
epochs = 1000
seed = 10
np.random.seed(seed)


class Classifier():
    def __init__(
        self,
        features_dim=36, output_dim=6, drop_out=False,
        keep_prob=0.9, num_hidden_layers=1,
        hidden_layer_dict={1: 10},
        batch_size=32, learning_rate=0.01,
        l2_beta=10**(-6), epochs=1000,
        early_stop=True, patience=20, min_delta=0.001,
        momentum=None,
        **kwargs
    ):

        self.features_dim = features_dim
        self.output_dim = output_dim
        self.drop_out = drop_out
        self.keep_prob = keep_prob
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_dict = hidden_layer_dict
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_beta = l2_beta
        self.epochs = epochs
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        if momentum:
            self.momentum = momentum
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
        #end if

        # accuracy
        self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.U, 1), tf.argmax(self.y_, 1)), tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)

        # cross-entropy
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.U))

        # loss
        self.loss = tf.reduce_mean(self.cross_entropy + self.l2_beta*self.regularization)

        # Create the gradient descent optimizer with the given learning rate.
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
    #end def

    def train(self, X_train, Y_train, X_test, Y_test, X_val, Y_val, **kwargs):
        np.random.seed(10)

        self.train_err = []
        self.test_acc = []
        N = len(X_train)
        idx = np.arange(N)
        time_to_update = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            tmp_best_val_err = 1000 # keep track of best_loss
            _patience = self.patience
            _epochs = 0
            for i in range(self.epochs):
                _epochs += 1
                np.random.shuffle(idx)
                X_train = X_train[idx]
                Y_train = Y_train[idx]

                t = time.time()
                for _start, _end in zip(range(0, N, self.batch_size), range(self.batch_size, N, self.batch_size)):
                    self.train_op.run(feed_dict={self.x: X_train[_start:_end], self.y_: Y_train[_start:_end]})
                time_to_update += (time.time() - t)

                self.train_err.append(self.loss.eval(feed_dict={self.x: X_train, self.y_: Y_train}))
                self.test_acc.append(self.accuracy.eval(feed_dict={self.x: X_test, self.y_: Y_test}))

                if self.early_stop:
                    _val_err = self.loss.eval(feed_dict={self.x: X_val, self.y_: Y_val})
                    if (tmp_best_val_err - _val_err) < self.min_delta:
                        _patience -= 1
                        if _patience == 0:
                            print('Early stopping at {}th iteration'.format(i))
                            print('-'*50)
                            break
                    else:
                        _patience = self.patience
                        tmp_best_val_err = _val_err
                    #end if
                #end if

                if i % 200 == 0:
                    print('batch size: %d: hidden neurons: [%d] decay parameters: %g iter: %d, train error  : %g'%(self.batch_size, self.hidden_layer_dict[1], self.l2_beta, i, self.train_err[i]))
                    print('batch size: %d: hidden neurons: [%d] decay parameters: %g iter: %d, test accuracy  : %g'%(self.batch_size, self.hidden_layer_dict[1], self.l2_beta, i, self.test_acc[i]))
                    print('-'*50)

            #end for
            self.early_stop_epoch = _epochs
            self.saver.save(sess, ".ckpt/1amodel.ckpt")
        #end with

        self.time_taken_one_epoch = (time_to_update/_epochs) * 1000
        return self
    #end def


    def test(self, X_test, Y_test):
        with tf.Session() as sess:
            self.saver.restore(sess, ".ckpt/1amodel.ckpt")
            test_error = self.loss.eval(feed_dict={self.x: X_test, self.y_: Y_test})
        #end with
        return test_error
    #end def


    def predict(self, X):
        with tf.Session() as sess:
            self.saver.restore(sess, ".ckpt/1amodel.ckpt")
            feed_dict = {self.x: X}
            prediction = tf.argmax(self.U, 1)
            prediction = prediction.eval(feed_dict=feed_dict)
        #end with

        prediction = [pred + 1 if pred != 5 else pred + 2 for pred in prediction]
        return prediction
    #end def
#end class


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)
#end def


def _read_data(file_name):
    _input = np.loadtxt(file_name, delimiter=' ')
    X, _Y = _input[:, :36], _input[:, -1].astype(int)
    _Y[_Y == 7] = 6

    # one hot matrix
    Y = np.zeros((_Y.shape[0], NUM_CLASSES))
    Y[np.arange(_Y.shape[0]), _Y-1] = 1
    return X, Y
#end def


def _transform_Y(Y):
    new_Y = []
    for y in Y:
        index = list(y).index(1.0)
        new_Y.append(index+1)

    new_Y = [7 if y == 6 else y for y in new_Y]
    return new_Y
#end def


def main():
    error_against_epoch = [0, 1000, 0, 2]
    accuracy_against_epoch = [0, 1000, 0, 1]
    es_error_against_epoch = [0, 1000, 0, 2]
    es_accuracy_against_epoch = [0, 1000, 0, 1]

    # read train data
    trainX, trainY = _read_data('./data/sat_train.txt')

    # read test data
    testX, testY = _read_data('./data/sat_test.txt')

    X_train, X_val, Y_train, Y_val = train_test_split(trainX, trainY, test_size=0.25, random_state=10)
    x_min = np.min(X_train, axis=0)
    x_max = np.max(X_train, axis=0)
    X_train = scale(X_train, x_min, x_max)
    X_val = scale(X_val, x_min, x_max)
    X_test = scale(testX, x_min, x_max)
    Y_test = testY

    train_test_val = dict(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, X_val=X_val, Y_val = Y_val)
    # =====================Q1 Design a ffn with one hidden layer=====================
    classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
                            hidden_layer_dict={1: 10},
                            ).train(**train_test_val)

    train_err1, test_acc1, time_taken_one_epoch1, early_stop_epoch = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch, classifier.early_stop_epoch

    # # plot train_err against epoch
    # plt.figure('Training Error: 1 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.title('Training Error')
    # plt.plot(range(epochs), train_err1)
    # plt.xlabel(str(epochs) + ' iterations')
    # plt.ylabel('Train Error')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/1a_train_error_with_3_layer_network.png')
    #
    # # plot test_acc against epoch
    # plt.figure('Test Accuracy: 1 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.title('Test Accuracy')
    # plt.plot(range(epochs), test_acc1)
    # plt.xlabel(str(epochs) + ' iterations')
    # plt.ylabel('Test Accuracy')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/1a_test_accuracy_with_3_layer_network.png')

    # # =====================Q2 Determine optimal batch size=====================
    batch_sizes = [4,8,16,32,64]

    #### With Early Stopping
    train_err_list = []
    test_acc_list = []
    time_taken_one_epoch_list = []
    early_stop_epoch_list = []
    predicted_dict = dict()

    for batch_size in batch_sizes:
        classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
                                hidden_layer_dict={1: 10}, batch_size=batch_size,
                                early_stop=True, patience=20, min_delta=0.001).train(**train_test_val)
        train_err, test_acc, time_taken_one_epoch, early_stop_epoch = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch, classifier.early_stop_epoch
        train_err_list.append(train_err)
        test_acc_list.append(test_acc)
        time_taken_one_epoch_list.append(time_taken_one_epoch)
        early_stop_epoch_list.append(early_stop_epoch)
        predicted_dict[batch_size] = classifier.predict(X_test)
    #end for

    # Plot Training Errors
    plt.figure("Early Stopping Train Error against Epoch with different Batch Sizes")
    plt.title("Early Stopping Train Error against Epoch with different Batch Sizes")
    es_error_against_epoch[1] = max([len(l) for l in train_err_list])
    plt.axis(es_error_against_epoch)
    for i in range(len(batch_sizes)):
        plt.plot(range(len(train_err_list[i])), train_err_list[i], label = 'batch_size = {}'.format(batch_sizes[i]))
        plt.xlabel('Epochs')
        plt.ylabel('Train Error')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.savefig('figures/1a/2a_es_train_error_vs_epoch_for_diff_batch_size.png')

    # Plot Test Accuracy
    plt.figure("Early Stopping Test Accuracy against Epoch with different Batch Sizes")
    plt.title("Early Stopping Test Accuracy against Epoch with different Batch Sizes")
    es_accuracy_against_epoch[1] = max([len(l) for l in test_acc_list])
    plt.axis(es_accuracy_against_epoch)
    for i in range(len(batch_sizes)):
        plt.plot(range(len(test_acc_list[i])), test_acc_list[i], label = 'batch_size = {}'.format(batch_sizes[i]))
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
    plt.savefig('figures/1a/2a_es_test_accuracy_vs_epoch_for_diff_batch_size.png')
    #end for

    # Plot Time Taken for One Epoch
    plt.figure("Early Stopping Time Taken for One Epoch against Batch Size")
    plt.title("Early Stopping Time Taken for One Epoch against Batch Size")
    plt.plot(batch_sizes, time_taken_one_epoch_list)
    plt.xlabel('Batch Size')
    plt.ylabel('Time/ms')
    plt.grid(b=True)
    plt.savefig('figures/1a/2b_es_time_taken_for_one_epoch_vs_batch_size.png')

    total_time_taken_list = [x*y for x,y in zip(early_stop_epoch_list,time_taken_one_epoch_list)]
    # Plot Total Time Taken
    plt.figure("Early Stopping Total Time Taken against Batch Size")
    plt.title("Early Stopping Total Time Taken against Batch Size")
    plt.plot(batch_sizes, total_time_taken_list)
    plt.xlabel('Batch Size')
    plt.ylabel('Time/ms')
    plt.grid(b=True)
    plt.savefig('figures/1a/2b_es_total_time_taken_vs_batch_size.png')

    #plot converged test accuracy against Number of Neurons
    final_acc = [acc[-1] for acc in test_acc_list]
    plt.figure('Early Stopping Converged Accuracy against Batch Size')
    plt.title('Early Stopping Converged Accuracy against Batch Size')
    plt.plot(batch_sizes, final_acc)
    plt.xlabel('Batch Size')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/1a/2c_es_converged_accuracy_against_batch_size.png')

    for i in range(len(batch_sizes)):
        print('Batch Size: {}'.format(batch_sizes[i]))
        print('Time per epoch: {}ms'.format(time_taken_one_epoch_list[i]))
        print('Total Time: {}ms'.format(total_time_taken_list[i]))
        print('Convergence Test Accuracy: {}'.format(final_acc[i]))
        print('-'*50)

    # --------------------------------------------------
    # Batch Size: 4
    # Time per epoch: 241.12526020582985ms
    # Total Time: 65586.07077598572ms
    # Convergence Test Accuracy: 0.8684999942779541
    # --------------------------------------------------
    # Batch Size: 8
    # Time per epoch: 125.91645615232503ms
    # Total Time: 41048.76470565796ms
    # Convergence Test Accuracy: 0.8560000061988831
    # --------------------------------------------------
    # Batch Size: 16
    # Time per epoch: 65.06782950776996ms
    # Total Time: 12883.430242538452ms
    # Convergence Test Accuracy: 0.8360000252723694
    # --------------------------------------------------
    # Batch Size: 32
    # Time per epoch: 34.805908510761874ms
    # Total Time: 10789.831638336182ms
    # Convergence Test Accuracy: 0.8309999704360962
    # --------------------------------------------------
    # Batch Size: 64
    # Time per epoch: 20.350156632144895ms
    # Total Time: 6898.703098297119ms
    # Convergence Test Accuracy: 0.8274999856948853
    # --------------------------------------------------

    # _transform_Y(predicted_dict[64])
    for batch_size in batch_sizes:
        print('Early Stopping Batch size {} Test set classification report:\n{}'.format(batch_size, classification_report(_transform_Y(Y_test), predicted_dict[batch_size], digits=3, labels=np.unique(predicted_dict[batch_size]))))

    # --------------------------------------------------
    #     Early Stopping Batch size 4 Test set classification report:
    #              precision    recall  f1-score   support
    #
    #           1      0.970     0.987     0.978       461
    #           2      0.956     0.969     0.962       224
    #           3      0.912     0.859     0.885       397
    #           4      0.542     0.768     0.635       211
    #           5      0.871     0.857     0.864       237
    #           7      0.902     0.764     0.827       470
    #
    # avg / total      0.884     0.869     0.873      2000
    #
    # Early Stopping Batch size 8 Test set classification report:
    #              precision    recall  f1-score   support
    #
    #           1      0.976     0.974     0.975       461
    #           2      0.952     0.964     0.958       224
    #           3      0.823     0.957     0.885       397
    #           4      0.598     0.318     0.415       211
    #           5      0.847     0.865     0.856       237
    #           7      0.795     0.840     0.817       470
    #
    # avg / total      0.845     0.856     0.845      2000
    #
    # Early Stopping Batch size 16 Test set classification report:
    #              precision    recall  f1-score   support
    #
    #           1      0.962     0.985     0.973       461
    #           2      0.942     0.942     0.942       224
    #           3      0.891     0.902     0.896       397
    #           4      0.461     0.308     0.369       211
    #           5      0.841     0.734     0.784       237
    #           7      0.740     0.872     0.801       470
    #
    # avg / total      0.826     0.836     0.828      2000
    #
    # Early Stopping Batch size 32 Test set classification report:
    #              precision    recall  f1-score   support
    #
    #           1      0.964     0.985     0.974       461
    #           2      0.941     0.929     0.935       224
    #           3      0.866     0.924     0.894       397
    #           4      0.463     0.265     0.337       211
    #           5      0.806     0.717     0.759       237
    #           7      0.737     0.866     0.796       470
    #
    # avg / total      0.817     0.831     0.819      2000
    #
    # Early Stopping Batch size 64 Test set classification report:
    #              precision    recall  f1-score   support
    #
    #           1      0.954     0.987     0.970       461
    #           2      0.941     0.929     0.935       224
    #           3      0.870     0.942     0.904       397
    #           4      0.469     0.251     0.327       211
    #           5      0.790     0.667     0.723       237
    #           7      0.728     0.866     0.791       470
    #
    # avg / total      0.812     0.828     0.814      2000

    optimal_batch_size = 4


    # =====================Q3 Determine optimal number of hidden neurons=====================
    num_hidden_neurons = [5,10,15,20,25]

    #### With Early Stopping
    train_err_list = []
    test_acc_list = []
    time_taken_one_epoch_list = []
    early_stop_epoch_list = []
    predicted_dict = dict()

    for num_neurons in num_hidden_neurons:
        classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
                                hidden_layer_dict={1: num_neurons}, batch_size=optimal_batch_size,
                                early_stop=True, patience=20, min_delta=0.001).train(**train_test_val)
        train_err, test_acc, time_taken_one_epoch, early_stop_epoch = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch, classifier.early_stop_epoch
        train_err_list.append(train_err)
        test_acc_list.append(test_acc)
        time_taken_one_epoch_list.append(time_taken_one_epoch)
        early_stop_epoch_list.append(early_stop_epoch)
        predicted_dict[num_neurons] = classifier.predict(X_test)
    #end for

    # Plot Training Errors
    plt.figure("Early Stopping Train Error against Epoch with different Number of Neurons")
    plt.title("Early Stopping Train Error against Epoch with different Number of Neurons")
    es_error_against_epoch[1] = max([len(l) for l in train_err_list])
    plt.axis(es_error_against_epoch)

    for i in range(len(num_hidden_neurons)):
        plt.plot(range(len(train_err_list[i])), train_err_list[i], label = 'num_neurons = {}'.format(num_hidden_neurons[i]))
        plt.xlabel('Epochs')
        plt.ylabel('Train Error')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.savefig('figures/1a/3a_es_train_error_vs_epoch_for_diff_num_neurons.png')


    # Plot Test Accuracy
    plt.figure("Early Stopping Test Accuracy against Epoch with different Number of Neurons")
    plt.title("Early Stopping Test Accuracy against Epoch with different Number of Neurons")
    es_accuracy_against_epoch[1] = max([len(l) for l in test_acc_list])
    plt.axis(es_accuracy_against_epoch)
    for i in range(len(num_hidden_neurons)):
        plt.plot(range(len(test_acc_list[i])), test_acc_list[i], label = 'num_neurons = {}'.format(num_hidden_neurons[i]))
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.savefig('figures/1a/3a_es_test_accuracy_vs_epoch_for_diff_num_neurons.png')


    # Plot Time Taken for One Epoch
    plt.figure("Early Stopping Time Taken for One Epoch against Number of Neurons")
    plt.title("Early Stopping Time Taken for One Epoch against Number of Neurons")
    plt.plot(num_hidden_neurons, time_taken_one_epoch_list)
    plt.xlabel('Number of Neurons')
    plt.ylabel('Time/ms')
    plt.grid(b=True)
    plt.savefig('figures/1a/3b_es_time_taken_for_one_epoch_vs_num_neurons.png')

    total_time_taken_list = [x*y for x,y in zip(early_stop_epoch_list,time_taken_one_epoch_list)]
    # Plot Total Time Taken
    plt.figure("Early Stopping Total Time Taken against Number of Neurons")
    plt.title("Early Stopping Total Time Taken against Number of Neurons")
    plt.plot(num_hidden_neurons, total_time_taken_list)
    plt.xlabel('Number of Neurons')
    plt.ylabel('Time/ms')
    plt.grid(b=True)
    plt.savefig('figures/1a/3b_es_total_time_taken_vs_num_neurons.png')

    # plot final test accuracy against Number of Neurons
    final_acc = [acc[-1] for acc in test_acc_list]
    plt.figure('Early Stopping Converged Accuracy against Number of Neurons')
    plt.title('Early Stopping Converged Accuracy against Number of Neurons')
    plt.plot(num_hidden_neurons, final_acc)
    plt.xlabel('Number of Neurons')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/1a/3c_es_accuracy_against_num_neurons.png')

    for i in range(len(num_hidden_neurons)):
        print('Number of Neurons: {}'.format(num_hidden_neurons[i]))
        print('Time per epoch: {}ms'.format(time_taken_one_epoch_list[i]))
        print('Total Time: {}ms'.format(total_time_taken_list[i]))
        print('Convergence Test Accuracy: {}'.format(final_acc[i]))
        print('-'*50)

    # --------------------------------------------------
    # Number of Neurons: 5
    # Time per epoch: 223.90767412448147ms
    # Total Time: 24405.93647956848ms
    # Convergence Test Accuracy: 0.847000002861023
    # --------------------------------------------------
    # Number of Neurons: 10
    # Time per epoch: 241.24092827824987ms
    # Total Time: 65617.53249168396ms
    # Convergence Test Accuracy: 0.8684999942779541
    # --------------------------------------------------
    # Number of Neurons: 15
    # Time per epoch: 248.12683029847ms
    # Total Time: 56324.79047775269ms
    # Convergence Test Accuracy: 0.8579999804496765
    # --------------------------------------------------
    # Number of Neurons: 20
    # Time per epoch: 254.77612436863413ms
    # Total Time: 58088.956356048584ms
    # Convergence Test Accuracy: 0.875
    # --------------------------------------------------
    # Number of Neurons: 25
    # Time per epoch: 253.72774500242423ms
    # Total Time: 54044.00968551636ms
    # Convergence Test Accuracy: 0.8615000247955322
    # --------------------------------------------------

    for num_neurons in num_hidden_neurons:
        print('Early Stopping Number of Neurons {} Test set classification report:\n{}'.format(num_neurons, classification_report(_transform_Y(Y_test), predicted_dict[num_neurons], digits=3, labels=np.unique(predicted_dict[num_neurons]))))

    # --------------------------------------------------
    # Early Stopping Number of Neurons 5 Test set classification report:
    #              precision    recall  f1-score   support

    #           1      0.952     0.991     0.971       461
    #           2      0.929     0.938     0.933       224
    #           3      0.910     0.869     0.889       397
    #           4      0.505     0.507     0.506       211
    #           5      0.830     0.802     0.815       237
    #           7      0.812     0.819     0.816       470

    # avg / total      0.847     0.847     0.847      2000


    # Early Stopping Number of Neurons 10 Test set classification report:
    #              precision    recall  f1-score   support

    #           1      0.970     0.987     0.978       461
    #           2      0.956     0.969     0.962       224
    #           3      0.912     0.859     0.885       397
    #           4      0.542     0.768     0.635       211
    #           5      0.871     0.857     0.864       237
    #           7      0.902     0.764     0.827       470

    # avg / total      0.884     0.869     0.873      2000

    # Early Stopping Number of Neurons 15 Test set classification report:
    #              precision    recall  f1-score   support

    #           1      0.966     0.989     0.977       461
    #           2      0.963     0.929     0.945       224
    #           3      0.894     0.909     0.901       397
    #           4      0.598     0.360     0.450       211
    #           5      0.833     0.861     0.846       237
    #           7      0.767     0.874     0.817       470

    # avg / total      0.850     0.858     0.850      2000

    # Early Stopping Number of Neurons 20 Test set classification report:
    #              precision    recall  f1-score   support

    #           1      0.985     0.983     0.984       461
    #           2      0.947     0.964     0.956       224
    #           3      0.869     0.937     0.902       397
    #           4      0.626     0.706     0.664       211
    #           5      0.871     0.823     0.846       237
    #           7      0.865     0.777     0.818       470

    # avg / total      0.878     0.875     0.875      2000

    # Early Stopping Number of Neurons 25 Test set classification report:
    #              precision    recall  f1-score   support

    #           1      0.976     0.983     0.979       461
    #           2      0.960     0.973     0.967       224
    #           3      0.860     0.947     0.902       397
    #           4      0.674     0.275     0.391       211
    #           5      0.918     0.802     0.856       237
    #           7      0.739     0.911     0.816       470

    # avg / total      0.857     0.862     0.847      2000

    optimal_num_neurons = 20

    # =====================Q4 Determine optimal decay parameter=====================
    beta_list = [0,1e-12,1e-9,1e-6,1e-3]

    #### With Early Stopping
    train_err_list = []
    test_acc_list = []
    time_taken_one_epoch_list = []
    early_stop_epoch_list = []
    predicted_dict = dict()

    for beta in beta_list:
        classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
                                hidden_layer_dict={1: optimal_num_neurons}, batch_size=optimal_batch_size,
                                l2_beta=beta, early_stop=True, patience=20, min_delta=0.001).train(**train_test_val)

        train_err, test_acc, time_taken_one_epoch, early_stop_epoch = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch, classifier.early_stop_epoch
        train_err_list.append(train_err)
        test_acc_list.append(test_acc)
        time_taken_one_epoch_list.append(time_taken_one_epoch)
        early_stop_epoch_list.append(early_stop_epoch)
        print('{} beta took {}ms per epoch'.format(beta, time_taken_one_epoch))
        predicted_dict[beta] = classifier.predict(X_test)

    # Plot Training Errors
    plt.figure("Early Stopping Train Error against Epoch with Different Decay Parameters")
    plt.title("Early Stopping Train Error against Epoch with Different Decay Parameters")
    es_error_against_epoch[1] = max([len(l) for l in train_err_list])
    plt.axis(es_error_against_epoch)

    for i in range(len(beta_list)):
        plt.plot(range(len(train_err_list[i])), train_err_list[i], label = 'beta = {}'.format(beta_list[i]))
        plt.xlabel('Epochs')
        plt.ylabel('Train Error')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.savefig('figures/1a/4a_es_train_error_vs_epoch_for_diff_beta.png')

    # Plot Test Accuracy
    plt.figure("Early Stopping Test Accuracy against Epoch with Different Decay Parameters")
    plt.title("Early Stopping Test Accuracy against Epoch with Different Decay Parameters")
    es_accuracy_against_epoch[1] = max([len(l) for l in test_acc_list])
    plt.axis(es_accuracy_against_epoch)
    for i in range(len(beta_list)):
        plt.plot(range(len(test_acc_list[i])), test_acc_list[i], label = 'beta = {}'.format(beta_list[i]))
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.savefig('figures/1a/4b_es_test_accuracy_vs_epoch_for_diff_beta.png')

    # Plot Time Taken for One Epoch
    plt.figure("Early Stopping Time Taken for One Epoch against Decay Parameters")
    plt.title("Early Stopping Time Taken for One Epoch against Decay Parametersa")
    plt.xticks(np.arange(5), [str(beta) for beta in beta_list])
    plt.plot([str(beta) for beta in beta_list], time_taken_one_epoch_list)
    plt.xlabel('Decay Parameters')
    plt.ylabel('Time/ms')
    plt.grid(b=True)
    plt.savefig('figures/1a/4b_es_time_taken_for_one_epoch_vs_diff_beta.png')

    total_time_taken_list = [x*y for x,y in zip(early_stop_epoch_list,time_taken_one_epoch_list)]
    # Plot Total Time Taken
    plt.figure("Early Stopping Total Time Taken against Number of Neurons")
    plt.title("Early Stopping Total Time Taken against Number of Neurons")
    plt.xticks(np.arange(5), [str(beta) for beta in beta_list])
    plt.plot([str(beta) for beta in beta_list], total_time_taken_list)
    plt.xlabel('Decay Parameters')
    plt.ylabel('Time/ms')
    plt.grid(b=True)
    plt.savefig('figures/1a/4b_es_total_time_taken_vs_diff_beta.png')

    # Plot Test Accuracy against Decay Parameters
    final_acc = [acc[-1] for acc in test_acc_list]
    plt.figure('Early Stopping Test Accuracy against Decay Parameters')
    plt.title('Early Stopping Test Accuracy against Decay Parameters')
    plt.xticks(np.arange(5), [str(beta) for beta in beta_list])
    plt.plot([str(beta) for beta in beta_list], final_acc)
    plt.xlabel('Decay Parameters')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/1a/4b_es_test_accuracy_against_decay_arameters.png')

    for i in range(len(beta_list)):
        print('Beta: {}'.format(beta_list[i]))
        print('Time per epoch: {}ms'.format(time_taken_one_epoch_list[i]))
        print('Total Time: {}ms'.format(total_time_taken_list[i]))
        print('Convergence Test Accuracy: {}'.format(final_acc[i]))
        print('-'*50)

    # --------------------------------------------------
    # Beta: 0
    # Time per epoch: 235.09231994026587ms
    # Total Time: 53601.048946380615ms
    # Convergence Test Accuracy: 0.8755000233650208
    # --------------------------------------------------
    # Beta: 1e-12
    # Time per epoch: 242.66248627712852ms
    # Total Time: 55327.0468711853ms
    # Convergence Test Accuracy: 0.8755000233650208
    # --------------------------------------------------
    # Beta: 1e-09
    # Time per epoch: 243.50313241021675ms
    # Total Time: 55518.71418952942ms
    # Convergence Test Accuracy: 0.875
    # --------------------------------------------------
    # Beta: 1e-06
    # Time per epoch: 241.57435538475974ms
    # Total Time: 55078.95302772522ms
    # Convergence Test Accuracy: 0.875
    # --------------------------------------------------
    # Beta: 0.001
    # Time per epoch: 242.8918645737019ms
    # Total Time: 45663.67053985596ms
    # Convergence Test Accuracy: 0.8514999747276306
    # --------------------------------------------------

    for beta in beta_list:
        print('Early Stopping beta {} Test set classification report:\n{}'.format(beta, classification_report(_transform_Y(Y_test), predicted_dict[beta], digits=3, labels=np.unique(predicted_dict[beta]))))

    #--------------------------------------------------
    # Early Stopping beta 0 Test set classification report:
    #              precision    recall  f1-score   support
    #
    #           1      0.983     0.983     0.983       461
    #           2      0.952     0.964     0.958       224
    #           3      0.871     0.937     0.903       397
    #           4      0.626     0.706     0.664       211
    #           5      0.871     0.823     0.846       237
    #           7      0.865     0.779     0.820       470
    #
    # avg / total      0.879     0.875     0.876      2000
    #
    # Early Stopping beta 1e-12 Test set classification report:
    #              precision    recall  f1-score   support
    #
    #           1      0.985     0.983     0.984       461
    #           2      0.952     0.964     0.958       224
    #           3      0.871     0.937     0.903       397
    #           4      0.623     0.706     0.662       211
    #           5      0.871     0.827     0.848       237
    #           7      0.865     0.777     0.818       470
    #
    # avg / total      0.879     0.875     0.876      2000
    #
    # Early Stopping beta 1e-09 Test set classification report:
    #              precision    recall  f1-score   support
    #
    #           1      0.985     0.983     0.984       461
    #           2      0.947     0.964     0.956       224
    #           3      0.869     0.937     0.902       397
    #           4      0.626     0.706     0.664       211
    #           5      0.871     0.823     0.846       237
    #           7      0.865     0.777     0.818       470
    #
    # avg / total      0.878     0.875     0.875      2000
    #
    # Early Stopping beta 1e-06 Test set classification report:
    #              precision    recall  f1-score   support
    #
    #           1      0.985     0.983     0.984       461
    #           2      0.947     0.964     0.956       224
    #           3      0.869     0.937     0.902       397
    #           4      0.626     0.706     0.664       211
    #           5      0.871     0.823     0.846       237
    #           7      0.865     0.777     0.818       470
    #
    # avg / total      0.878     0.875     0.875      2000
    #
    # Early Stopping beta 0.001 Test set classification report:
    #              precision    recall  f1-score   support
    #
    #           1      0.962     0.991     0.976       461
    #           2      0.963     0.938     0.950       224
    #           3      0.896     0.894     0.895       397
    #           4      0.543     0.512     0.527       211
    #           5      0.856     0.751     0.800       237
    #           7      0.784     0.840     0.811       470
    #
    # avg / total      0.850     0.852     0.850      2000

    optimal_beta = 1e-6

    # =====================Q5=====================

    #### With Early Stopping
    train_err_list = []
    test_acc_list = []
    time_taken_one_epoch_list =[]
    early_stop_epoch_list = []

    #### 3-layer With Early Stopping
    classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
                            hidden_layer_dict={1: 10}, early_stop=True, patience=20, min_delta=0.001
                            ).train(**train_test_val)

    train_err1, test_acc1, time_taken_one_epoch1, early_stop_epoch1 = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch, classifier.early_stop_epoch

    train_err_list.append(train_err1)
    test_acc_list.append(test_acc1)
    time_taken_one_epoch_list.append(time_taken_one_epoch1)
    early_stop_epoch_list.append(early_stop_epoch1)
    print('Early Stopping 3-layer took {}ms per epoch'.format(time_taken_one_epoch1))
    # Early Stopping 3-layer took 36.41304662150721ms per epoch
    print('Early Stopping 3-layer has converged test accuracy {}'.format(test_acc1[-1]))
    # Early Stopping 3-layer has converged test accuracy 0.83139908

    predicted_y = classifier.predict(X_test)

    print('Early Stopping 3-layer Test set classification report:\n{}'.format(classification_report(_transform_Y(Y_test), predicted_y, digits=3, labels=np.unique(predicted_y))))

    # Early Stopping 3-layer Test set classification report:
    #              precision    recall  f1-score   support

    #           1      0.964     0.985     0.974       461
    #           2      0.941     0.929     0.935       224
    #           3      0.866     0.924     0.894       397
    #           4      0.463     0.265     0.337       211
    #           5      0.806     0.717     0.759       237
    #           7      0.737     0.866     0.796       470

    # avg / total      0.817     0.831     0.819      2000


    #### 4-layer With Early Stopping
    classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
                            hidden_layer_dict={1: 10, 2: 10}, num_hidden_layers=2,
                            early_stop=True, patience=20, min_delta=0.001).train(**train_test_val)

    train_err2, test_acc2, time_taken_one_epoch2, early_stop_epoch2 = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch, classifier.early_stop_epoch
    print('Early stopping 4-layer took {}ms per epoch'.format(time_taken_one_epoch2))
    # Early stopping 4-layer took 43.69016647338867ms per epoch

    print('Early Stopping 4-layer has converged test accuracy {}'.format(test_acc1[-1]))
    # Early Stopping 4-layer has converged test accuracy 0.847806713

    predicted_y = classifier.predict(X_test)

    print('Early Stopping 4-layer Test set classification report:\n{}'.format(classification_report(_transform_Y(Y_test), predicted_y, digits=3, labels=np.unique(predicted_y))))

    # Early Stopping 4-layer Test set classification report:
    #              precision    recall  f1-score   support

    #           1      0.954     0.987     0.970       461
    #           2      0.950     0.938     0.944       224
    #           3      0.854     0.942     0.896       397
    #           4      0.542     0.398     0.459       211
    #           5      0.847     0.768     0.805       237
    #           7      0.791     0.832     0.811       470

    # avg / total      0.839     0.848     0.842      2000

    train_err_list.append(train_err2)
    test_acc_list.append(test_acc2)
    time_taken_one_epoch_list.append(time_taken_one_epoch2)
    early_stop_epoch_list.append(early_stop_epoch2)

    # plot train_err against epoch
    plt.figure('Early Stopping Training Error: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    plt.title('Early Stopping Training Error: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    plt.plot(range(len(train_err2)), train_err2)
    plt.xlabel('Epochs')
    plt.ylabel('Train Error')
    plt.grid(b=True)
    plt.savefig('figures/1a/5a_es_train_error_with_4_layer_network.png')

    # plot test_acc against epoch
    plt.figure('Early Stopping Test Accuracy: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    plt.title('Early Stopping Test Accuracy: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    plt.plot(range(len(test_acc2)), test_acc2)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/1a/5a_es_test_accuracy_with_4_layer_network.png')

    # Q5(2) With Stopping
    # Plot Training Errors
    plt.figure("Early Stopping Train Error against Epoch with Different Number of Hidden Layers")
    plt.title("Early Stopping Train Error against Epoch with Different Number of Hidden Layers")
    es_error_against_epoch[1] = max([len(l) for l in train_err_list])
    plt.axis(es_error_against_epoch)

    for i in range(len(train_err_list)):
        plt.plot(range(len(train_err_list[i])), train_err_list[i], label = 'Number of Hidden Layers = {}'.format(i+1))
        plt.xlabel('Epochs')
        plt.ylabel('Train Error')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.savefig('figures/1a/5b_es_train_error_vs_epoch_for_diff_num_hidden_layers.png')

    # Plot Test Accuracy
    plt.figure("Early Stopping Test Accuracy against Epoch with Different Number of Hidden Layers")
    plt.title("Early Stopping Test Accuracy against Epoch with Different Number of Hidden Layers")
    es_accuracy_against_epoch[1] = max([len(l) for l in test_acc_list])
    plt.axis(es_accuracy_against_epoch)

    for i in range(len(test_acc_list)):
        plt.plot(range(len(test_acc_list[i])), test_acc_list[i], label = 'Number of Hidden Layers = {}'.format(i+1))
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.savefig('figures/1a/5b_es_test_accuracy_vs_epoch_for_diff_num_hidden_layers.png')

    # Plot Test Accuracy against num hidden layers
    final_acc = [acc[-1] for acc in test_acc_list]
    plt.figure('Early Stopping Converged Test Accuracy against Num Hidden Layers')
    plt.title('Early Stopping Converged Test Accuracy against Num Hidden Layers')
    plt.xticks(np.arange(2), ['1-hidden layer', '2-hidden layer'])
    plt.plot(['1-hidden layer', '2-hidden layer'], final_acc)
    plt.xlabel('Number of Hidden Layer')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/1a/5b_es_test_accuracy_against_num_hidden_layers.png')

    total_time_taken_list = [x*y for x,y in zip(early_stop_epoch_list,time_taken_one_epoch_list)]
    print ("Early Stopping Time Taken per Epoch for 3_layer: %g \nTime Taken per Epoch for 4_layer: %g" % (time_taken_one_epoch_list[0],time_taken_one_epoch_list[1]))
    # Early Stopping Time Taken per Epoch for 3_layer: 36.413
    # Time Taken per Epoch for 4_layer: 43.6902

    print ("Early Stopping Time taken for 3_layer: %g \nTime taken for 4_layer: %g" % (total_time_taken_list[0],total_time_taken_list[1]))
    # Early Stopping Time taken for 3_layer: 11288
    # Time taken for 4_layer: 8738.03

# end def

if __name__ == '__main__': main()
