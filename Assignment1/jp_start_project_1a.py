#
# Project 1, starter code part a
#
import logging
import math
import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn.metrics import classification_report
import time
import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')


logger = logging.getLogger(__name__)
NUM_FEATURES = 36
NUM_CLASSES = 6

# learning_rate = 0.01
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

        self._build_model()
    #end def

    
    def _build_layer(self, X, input_dim, output_dim, hidden=False):
        W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
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
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.U)

        # error
        self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.U))
       
        # loss
        self.loss = tf.reduce_mean(self.cross_entropy + self.l2_beta*self.regularization)

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
    #end def

    def train(self, trainX, trainY, testX, testY, small=False, **kwargs):

        if small:
            trainX = trainX[:300]
            trainY = trainY[:300]

        self.train_err = []
        self.train_acc = []
        self.test_acc = []

        N = len(trainX)
        idx = np.arange(N)
        time_to_update = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(self.epochs):
                np.random.shuffle(idx)
                X_train = trainX[idx]
                Y_train = trainY[idx]
                
                t = time.time()
                for _start, _end in zip(range(0, N, self.batch_size), range(self.batch_size, N, self.batch_size)):
                    self.train_op.run(feed_dict={self.x: X_train[_start:_end], self.y_: Y_train[_start:_end]})
                time_to_update += (time.time() - t)
  
                self.train_err.append(self.error.eval(feed_dict={self.x: X_train, self.y_: Y_train}))
                self.train_acc.append(self.accuracy.eval(feed_dict={self.x: X_train, self.y_: Y_train}))
                self.test_acc.append(self.accuracy.eval(feed_dict={self.x: testX, self.y_: testY}))

                if i % 100 == 0:
                    print('batch size: %d: hidden neurons: [%d] decay parameters: %g iter: %d, train error  : %g'%(self.batch_size, self.hidden_layer_dict[1], self.l2_beta, i, self.train_err[i]))
                    print('batch size: %d: hidden neurons: [%d] decay parameters: %g iter: %d, test accuracy  : %g'%(self.batch_size, self.hidden_layer_dict[1], self.l2_beta, i, self.test_acc[i]))
                    print('-'*50)
            #end for
        #end with

        self.time_taken_one_epoch = (time_to_update/epochs) * 1000
        return self
    #end def


    def test(self, X_test, Y_test):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            test_error = self.error.eval(feed_dict={self.x: X_test, self.y_: Y_test})
        #end with

        return test_error
    #end def


    def predict(self, X):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            feed_dict = {self.x: X}
            prediction = self.U.eval(feed_dict)
        #end with

        prediction = [[1.0 if _pred >= max(pred) else 0.0 for _pred in pred] for pred in prediction]
        return np.array(prediction)
    #end def
#end class


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)
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


def _transform_Y(Y):
    new_Y = []
    for y in Y:
        index = list(y).index(1.0)
        new_Y.append(index+1)
    
    new_Y = [7 if y == 6 else y for y in new_Y]
    return new_Y
#end def


def main():
    # read train data
    trainX, trainY, x_min, x_max = _read_data('sat_train.txt', train=True)

    # read test data
    testX, testY = _read_data('sat_test.txt', x_min=x_min, x_max=x_max)

    train_test = dict(trainX=trainX, trainY=trainY, testX=testX, testY=testY)
    # # =====================Q1 Design a ffn with one hidden layer=====================
    # classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
    #                         hidden_layer_dict={1: 10}).train(small=True, **train_test)

    # train_err1, train_acc1, test_acc1, time_taken_one_epoch1 = classifier.train_err, classifier.train_acc, classifier.test_acc, classifier.time_taken_one_epoch
    
    # # plot train_err against epoch
    # plt.figure('Training Error: 1 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.title('Training Error')
    # plt.plot(range(epochs), train_err1)
    # plt.xlabel(str(epochs) + ' iterations')
    # plt.ylabel('Train Accuracy')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/1a_train_error_with_3_layer_network.png')

    # # plot test_acc against epoch
    # plt.figure('Test Accuracy: 1 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.title('Test Accuracy')
    # plt.plot(range(epochs), test_acc1)
    # plt.xlabel(str(epochs) + ' iterations')
    # plt.ylabel('Test Accuracy')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/1a_test_accuracy_with_3_layer_network.png')


    # =====================Q2 Determine optimal batch size=====================

    # plot learning curves
    # batch_sizes = [4,8,16,32,64]
    batch_sizes = [64]

    train_err_list = []
    test_acc_list = []
    time_taken_one_epoch_list = []
    predicted_dict = dict()

    for batch_size in batch_sizes:
        classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
                                hidden_layer_dict={1: 10}, batch_size=batch_size).train(small=False, **train_test)
        train_err, test_acc, time_taken_one_epoch = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch
        train_err_list.append(train_err)
        test_acc_list.append(test_acc)
        time_taken_one_epoch_list.append(time_taken_one_epoch)
        predicted_dict[batch_size] = classifier.predict(testX)
    #end for

    # Plot Training Errors
    plt.figure("Train Error against Epoch with different Batch Sizes")
    plt.title("Train Error against Epoch with different Batch Sizes")
    for i in range(len(batch_sizes)):
        plt.plot(range(epochs), train_err_list[i], label = 'batch_size = {}'.format(batch_sizes[i]))
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Train Error')
        plt.legend()
        plt.grid(b=True)
        plt.savefig('figures/1a/2a_train_error_vs_epoch_for_diff_batch_size.png')
    #end for

    # Plot Test Accuracy
    plt.figure("Test Accuracy against Epoch with different Batch Sizes")
    plt.title("Test Accuracy against Epoch with different Batch Sizes")
    for i in range(len(batch_sizes)):
        plt.plot(range(epochs), test_acc_list[i], label = 'batch_size = {}'.format(batch_sizes[i]))
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
        plt.savefig('figures/1a/2a_test_accuracy_vs_epoch_for_diff_batch_size.png')
    #end for

    # Plot Time Taken for One Epoch
    plt.figure("Time Taken for One Epoch againt Batch Size")
    plt.title("Time Taken for One Epoch againt Batch Size")
    plt.plot(batch_sizes, time_taken_one_epoch_list)
    plt.xlabel('Batch Size')
    plt.ylabel('Time/ms')
    plt.grid(b=True)
    plt.savefig('figures/1a/2b_time_taken_for_one_epoch_vs_batch_size.png')

    #plot converged test accuracy against Number of Neurons
    final_acc = [acc[-1] for acc in test_acc_list]
    plt.figure('Accuracy against Batch Size')
    plt.title('Accuracy against Batch Size')
    plt.plot(batch_sizes, final_acc)
    plt.xlabel('Batch Size')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/1a/2c_Converged Accuracy against Batch Size.png')

    target_names = ['1', '2', '3', '4', '5', '7']
    for batch_size in batch_sizes:
        # print(_transform_Y(testY))
        # print(_transform_Y(predicted_dict[batch_size]))
        print('Batch size {} Test set classification report:\n{}'.format(batch_size, classification_report(_transform_Y(testY), _transform_Y(predicted_dict[batch_size]), digits=3, labels=np.unique(_transform_Y(predicted_dict[batch_size])))))

    # optimal_batch_size = 32

    # # =====================Q3 Determine optimal number of hidden neurons=====================

    # # plot learning curves
    # num_hidden_neurons = [5,10,15,20,25]
    # train_err_list = []
    # test_acc_list = []
    # time_taken_one_epoch_list = []

    # for num_neurons in num_hidden_neurons:
    #     classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
    #                             hidden_layer_dict={1: num_neurons}, batch_size=optimal_batch_size).train(small=True, **train_test)
    #     train_err, test_acc, time_taken_one_epoch = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch
    #     train_err_list.append(train_err)
    #     test_acc_list.append(test_acc)
    #     time_taken_one_epoch_list.append(time_taken_one_epoch)

    # # Plot Training Errors
    # plt.figure("Train Error against Epoch with different Number of Neurons")
    # plt.title("Train Error against Epoch with different Number of Neurons")
    # for i in range(len(num_hidden_neurons)):
    #     plt.plot(range(epochs), train_err_list[i], label = 'num_neurons = {}'.format(num_hidden_neurons[i]))
    #     plt.xlabel(str(epochs) + ' iterations')
    #     plt.ylabel('Train Error')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/3aa_train_error_vs_epoch_for_diff_num_neurons.png')

    # # Plot Test Accuracy
    # plt.figure("Test Accuracy against Epoch with different Number of Neurons")
    # plt.title("Test Accuracy against Epoch with different Number of Neurons")
    # for i in range(len(num_hidden_neurons)):
    #     plt.plot(range(epochs), test_acc_list[i], label = 'num_neurons = {}'.format(num_hidden_neurons[i]))
    #     plt.xlabel(str(epochs) + ' iterations')
    #     plt.ylabel('Test Accuracy')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/3a_test_accuracy_vs_epoch_for_diff_num_neurons.png')

    # # Plot Time Taken for One Epoch
    # plt.figure("Time Taken for One Epoch againt Number of Neurons")
    # plt.title("Time Taken for One Epoch againt Number of Neurons")
    # plt.plot(num_hidden_neurons, time_taken_one_epoch_list)
    # plt.xlabel('Number of Neurons')
    # plt.ylabel('Time/ms')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/3b_time_taken_for_one_epoch_vs_num_neurons.png')

    # # plot final test accuracy against Number of Neurons
    # final_acc = [acc[-1] for acc in test_acc_list]
    # plt.figure('Accuracy against Number of Neurons')
    # plt.title('Accuracy against Number of Neurons')
    # plt.plot(num_hidden_neurons, final_acc)
    # plt.xlabel('Number of Neurons')
    # plt.ylabel('Test Accuracy')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/3c_Accuracy against Number of Neurons.png')

    # optimal_num_neurons = 25

    # # =====================Q4 Determine optimal decay parameter=====================

    # # plot learning curves
    # beta_list = [0,1e-12,1e-9,1e-6,1e-3]
    # train_err_list = []
    # test_acc_list = []
    # time_taken_one_epoch_list = []

    # for beta in beta_list:
    #     classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
    #                             hidden_layer_dict={1: optimal_num_neurons}, batch_size=optimal_batch_size,
    #                             l2_beta=beta).train(small=True, **train_test)
    #     train_err, test_acc, time_taken_one_epoch = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch
    #     train_err_list.append(train_err)
    #     test_acc_list.append(test_acc)
    #     time_taken_one_epoch_list.append(time_taken_one_epoch)

    # # Plot Training Errors
    # plt.figure("Train Error against Epoch with Different Decay Parameters")
    # plt.title("Train Error against Epoch with Different Decay Parameters")
    # for i in range(len(beta_list)):
    #     plt.plot(range(epochs), train_err_list[i], label = 'beta = {}'.format(beta_list[i]))
    #     plt.xlabel(str(epochs) + ' iterations')
    #     plt.ylabel('Train Error')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/4a_train_error_vs_epoch_for_diff_beta.png')

    # # Plot Test Accuracy against Decay Parameters
    # final_acc = [acc[-1] for acc in test_acc_list]
    # plt.figure('Test Accuracy against Decay Parameters')
    # plt.title('Test Accuracy against Decay Parameters')
    # plt.xticks(np.arange(5), [str(beta) for beta in beta_list])
    # plt.plot([str(beta) for beta in beta_list], final_acc)
    # plt.xlabel('Decay Parameters')
    # plt.ylabel('Test Accuracy')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/4b_Test Accuracy against Decay Parameters.png')

    # optimal_beta = 0

    # # =====================Q5=====================

    # train_err_list = []
    # test_acc_list = []
    # time_taken_one_epoch_list =[]
    # train_err_list.append(train_err1)
    # test_acc_list.append(test_acc1)
    # time_taken_one_epoch_list.append(time_taken_one_epoch1)

    # classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
    #                         hidden_layer_dict={1: 10, 2: 10}, num_hidden_layers=2).train(small=True, **train_test)

    # train_err2, train_acc2, test_acc2, time_taken_one_epoch2 = classifier.train_err, classifier.train_acc, classifier.test_acc, classifier.time_taken_one_epoch

    # train_err_list.append(train_err2)
    # test_acc_list.append(test_acc2)
    # time_taken_one_epoch_list.append(time_taken_one_epoch2)

    # # plot train_err against epoch
    # plt.figure('Training Error: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.title('Training Error: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.plot(range(epochs), train_err2)
    # plt.xlabel(str(epochs) + ' iterations')
    # plt.ylabel('Train Error')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/5a_train_error_with_4_layer_network.png')

    # # plot train_acc against epoch
    # plt.figure('Training Accuracy: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.title('Training Accuracy: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.plot(range(epochs), train_acc2)
    # plt.xlabel(str(epochs) + ' iterations')
    # plt.ylabel('Train Accuracy')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/5a_train_accuracy_with_4_layer_network.png')

    # # plot test_acc against epoch
    # plt.figure('Test Accuracy: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.title('Test Accuracy: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.plot(range(epochs), test_acc2)
    # plt.xlabel(str(epochs) + ' iterations')
    # plt.ylabel('Test Accuracy')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/5a_test_accuracy_with_4_layer_network.png')

    # # Q5(2)
    # # Plot Training Errors
    # plt.figure("Train Error against Epoch with Different Number of Hidden Layers")
    # plt.title("Train Error against Epoch with Different Number of Hidden Layers")
    # for i in range(len(train_err_list)):
    #     plt.plot(range(epochs), train_err_list[i], label = 'Number of Hidden Layers = {}'.format(i+1))
    #     plt.xlabel(str(epochs) + ' iterations')
    #     plt.ylabel('Train Error')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/5b_train_error_vs_epoch_for_diff_num_hidden_layers.png')

    # # Plot Test Accuracy
    # plt.figure("Test Accuracy against Epoch with Different Number of Hidden Layers")
    # plt.title("Test Accuracy against Epoch with Different Number of Hidden Layers")
    # for i in range(len(test_acc_list)):
    #     plt.plot(range(epochs), test_acc_list[i], label = 'Number of Hidden Layers = {}'.format(i+1))
    #     plt.xlabel(str(epochs) + ' iterations')
    #     plt.ylabel('Test Accuracy')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/5b_test_accuracy_vs_epoch_for_diff_num_hidden_layers.png')

    # print ("Time taken for 3_layer: %g \nTime taken for 4_layer: %g" % (time_taken_one_epoch_list[0],time_taken_one_epoch_list[1]))
#end def

if __name__ == '__main__': main()
