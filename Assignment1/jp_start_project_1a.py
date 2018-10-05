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
        early_stop=False, patience=20, min_delta=0.005,
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
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.U)

        # # error
        # self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.U))
       
        # loss
        self.loss = tf.reduce_mean(self.cross_entropy + self.l2_beta*self.regularization)

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
    #end def

    def train(self, trainX, trainY, testX, testY, small=False, **kwargs):
        np.random.seed(10)

        if small:
            trainX = trainX[:100]
            trainY = trainY[:100]
        if self.early_stop:
            X_train, X_val, Y_train, Y_val = train_test_split(trainX, trainY, test_size=0.25, random_state=10)
        else:
            X_train, Y_train = trainX, trainY

        self.train_err = []
        self.val_err = []
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
                self.test_acc.append(self.accuracy.eval(feed_dict={self.x: testX, self.y_: testY}))
                
                # if i % 200 == 0:
                #     print('batch size: %d: hidden neurons: [%d] decay parameters: %g iter: %d, train error  : %g'%(self.batch_size, self.hidden_layer_dict[1], self.l2_beta, i, self.train_err[i]))
                #     print('batch size: %d: hidden neurons: [%d] decay parameters: %g iter: %d, test accuracy  : %g'%(self.batch_size, self.hidden_layer_dict[1], self.l2_beta, i, self.test_acc[i]))
                #     print('-'*50)

                if self.early_stop:
                    _val_err = self.loss.eval(feed_dict={self.x: X_val, self.y_: Y_val})
                    self.val_err.append(_val_err)

                    if (tmp_best_val_err - _val_err) < self.min_delta:
                        _patience -= 1
                        if _patience == 0:
                            print('Early stopping at {}th iteration'.format(i))
                            break
                    else: # if (tmp_best_val_err - _val_err) >= self.min_delta
                        _patience = self.patience
                        tmp_best_val_err = _val_err
                    #end if
                #end if
            #end for
            self.saver.save(sess, ".ckpt/1amodel.ckpt")
        #end with

        self.time_taken_one_epoch = (time_to_update/_epochs) * 1000
        return self
    #end def


    def test(self, X_test, Y_test):
        with tf.Session() as sess:
            self.saver.restore(sess, ".ckpt/1amodel.ckpt")

            test_error = self.error.eval(feed_dict={self.x: X_test, self.y_: Y_test})
        #end with

        return test_error
    #end def


    def predict(self, X):
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, ".ckpt/1amodel.ckpt")
            feed_dict = {self.x: X}
            prediction = tf.argmax(self.U, 1)
            prediction = prediction.eval(feed_dict=feed_dict)
        print(prediction)
        #end with

        prediction = [pred + 1 if pred != 5 else pred + 2 for pred in prediction]
        return prediction
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
    error_against_epoch = [0, 1000, 0, 2]
    accuracy_against_epoch = [0, 1000, 0, 1]
    es_error_against_epoch = [0, 1000, 0, 2]
    es_accuracy_against_epoch = [0, 1000, 0, 1]

    # read train data
    trainX, trainY, x_min, x_max = _read_data('./data/sat_train.txt', train=True)

    # read test data
    testX, testY = _read_data('./data/sat_test.txt', x_min=x_min, x_max=x_max)

    train_test = dict(trainX=trainX, trainY=trainY, testX=testX, testY=testY)
    # # =====================Q1 Design a ffn with one hidden layer===================== 
    # classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
    #                         hidden_layer_dict={1: 10},
    #                         ).train(small=False, **train_test)

    # train_err1, test_acc1, time_taken_one_epoch1 = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch
    
    # # plot train_err against epoch
    # plt.figure('Training Error: 1 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.title('Training Error')
    # plt.plot(range(epochs), train_err1)
    # plt.xlabel(str(epochs) + ' iterations')
    # plt.ylabel('Train Error')
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


    # # =====================Q2 Determine optimal batch size=====================
    # batch_sizes = [4,8,16,32,64]

    # #### Without Early Stopping 
    # train_err_list = []
    # test_acc_list = []
    # time_taken_one_epoch_list = []
    # predicted_dict = dict()

    # for batch_size in batch_sizes:
    #     classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
    #                             hidden_layer_dict={1: 10}, batch_size=batch_size).train(small=False, **train_test)
    #     train_err, test_acc, time_taken_one_epoch = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch
    #     train_err_list.append(train_err)
    #     test_acc_list.append(test_acc)
    #     time_taken_one_epoch_list.append(time_taken_one_epoch)
    #     print('{} batch size took {}ms per epoch'.format(batch_size, time_taken_one_epoch))
    #     predicted_dict[batch_size] = classifier.predict(testX)
    # #end for

    # # Plot Training Errors
    # plt.figure("Train Error against Epoch with different Batch Sizes")
    # plt.title("Train Error against Epoch with different Batch Sizes")
    # plt.axis(error_against_epoch)
    # for i in range(len(batch_sizes)):
    #     plt.plot(range(epochs), train_err_list[i], label = 'batch_size = {}'.format(batch_sizes[i]))
    #     plt.xlabel(str(epochs) + ' Epochs')
    #     plt.ylabel('Train Error')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/2a_train_error_vs_epoch_for_diff_batch_size.png')
    # #end for

    # # Plot Test Accuracy
    # plt.figure("Test Accuracy against Epoch with different Batch Sizes")
    # plt.title("Test Accuracy against Epoch with different Batch Sizes")
    # plt.axis(accuracy_against_epoch)
    # for i in range(len(batch_sizes)):
    #     plt.plot(range(epochs), test_acc_list[i], label = 'batch_size = {}'.format(batch_sizes[i]))
    #     plt.xlabel(str(epochs) + ' Epochs')
    #     plt.ylabel('Test Accuracy')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/2a_test_accuracy_vs_epoch_for_diff_batch_size.png')
    # #end for

    # # Plot Time Taken for One Epoch
    # plt.figure("Time Taken for One Epoch againt Batch Size")
    # plt.title("Time Taken for One Epoch againt Batch Size")
    # plt.plot(batch_sizes, time_taken_one_epoch_list)
    # plt.xlabel('Batch Size')
    # plt.ylabel('Time/ms')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/2b_time_taken_for_one_epoch_vs_batch_size.png')

    # #plot converged test accuracy against Number of Neurons
    # final_acc = [acc[-1] for acc in test_acc_list]
    # plt.figure('Converged Accuracy against Batch Size')
    # plt.title('Converged Accuracy against Batch Size')
    # plt.plot(batch_sizes, final_acc)
    # plt.xlabel('Batch Size')
    # plt.ylabel('Test Accuracy')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/2c_Converged Accuracy against Batch Size.png')

    # # _transform_Y(predicted_dict[64])
    # for batch_size in batch_sizes:
    #     print('Batch size {} Test set classification report:\n{}'.format(batch_size, classification_report(_transform_Y(testY), predicted_dict[batch_size], digits=3, labels=np.unique(predicted_dict[batch_size]))))

    # # Batch size 4 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.982     0.974     0.978       461
    # #           2      0.947     0.960     0.953       224
    # #           3      0.897     0.924     0.911       397
    # #           4      0.659     0.640     0.649       211
    # #           5      0.848     0.869     0.858       237
    # #           7      0.858     0.838     0.848       470
       
    # # avg / total      0.882     0.883     0.883      2000
       
    # # Batch size 8 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.980     0.967     0.974       461
    # #           2      0.956     0.960     0.958       224
    # #           3      0.887     0.929     0.908       397
    # #           4      0.685     0.649     0.667       211
    # #           5      0.850     0.861     0.855       237
    # #           7      0.864     0.853     0.859       470
       
    # # avg / total      0.885     0.886     0.885      2000
       
    # # Batch size 16 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.985     0.972     0.978       461
    # #           2      0.950     0.942     0.946       224
    # #           3      0.870     0.929     0.899       397
    # #           4      0.697     0.502     0.584       211
    # #           5      0.825     0.873     0.848       237
    # #           7      0.821     0.866     0.843       470
       
    # # avg / total      0.870     0.874     0.870      2000
       
    # # Batch size 32 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.982     0.970     0.976       461
    # #           2      0.959     0.929     0.943       224
    # #           3      0.872     0.927     0.899       397
    # #           4      0.634     0.460     0.533       211
    # #           5      0.806     0.861     0.833       237
    # #           7      0.800     0.851     0.825       470
       
    # # avg / total      0.857     0.862     0.858      2000

    # # Batch size 64 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.978     0.976     0.977       461
    # #           2      0.950     0.938     0.944       224
    # #           3      0.856     0.932     0.893       397
    # #           4      0.521     0.346     0.416       211
    # #           5      0.830     0.806     0.818       237
    # #           7      0.774     0.851     0.811       470
       
    # # avg / total      0.837     0.847     0.839      2000

    # #### With Early Stopping 
    # train_err_list = []
    # test_acc_list = []
    # time_taken_one_epoch_list = []
    # predicted_dict = dict()

    # for batch_size in batch_sizes:
    #     classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
    #                             hidden_layer_dict={1: 10}, batch_size=batch_size,
    #                             early_stop=True, patience=20, min_delta=0.005).train(small=False, **train_test)
    #     train_err, test_acc, time_taken_one_epoch = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch
    #     train_err_list.append(train_err)
    #     test_acc_list.append(test_acc)
    #     time_taken_one_epoch_list.append(time_taken_one_epoch)
    #     print('{} batch size took {}ms per epoch'.format(batch_size, time_taken_one_epoch))
    #     predicted_dict[batch_size] = classifier.predict(testX)
    # #end for

    # # Plot Training Errors
    # plt.figure("Early Stopping Train Error against Epoch with different Batch Sizes")
    # plt.title("Early Stopping Train Error against Epoch with different Batch Sizes")
    
    # es_error_against_epoch[1] = max([len(l) for l in train_err_list])
    # plt.axis(es_error_against_epoch)
    # for i in range(len(batch_sizes)):
    #     plt.plot(range(len(train_err_list[i])), train_err_list[i], label = 'batch_size = {}'.format(batch_sizes[i]))
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Train Error')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/2a_es_train_error_vs_epoch_for_diff_batch_size.png')
    # #end for

    # # Plot Test Accuracy
    # plt.figure("Early Stopping Test Accuracy against Epoch with different Batch Sizes")
    # plt.title("Early Stopping Test Accuracy against Epoch with different Batch Sizes")
   
    # es_accuracy_against_epoch[1] = max([len(l) for l in test_acc_list])
    # plt.axis(es_accuracy_against_epoch)
    # for i in range(len(batch_sizes)):
    #     plt.plot(range(len(test_acc_list[i])), test_acc_list[i], label = 'batch_size = {}'.format(batch_sizes[i]))
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Test Accuracy')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/2a_es_test_accuracy_vs_epoch_for_diff_batch_size.png')
    # #end for

    # # Plot Time Taken for One Epoch
    # plt.figure("Early Stopping Time Taken for One Epoch againt Batch Size")
    # plt.title("Early Stopping Time Taken for One Epoch againt Batch Size")
    # plt.plot(batch_sizes, time_taken_one_epoch_list)
    # plt.xlabel('Batch Size')
    # plt.ylabel('Time/ms')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/2b_es_time_taken_for_one_epoch_vs_batch_size.png')

    # #plot converged test accuracy against Number of Neurons
    # final_acc = [acc[-1] for acc in test_acc_list]
    # plt.figure('Early Stopping Converged Accuracy against Batch Size')
    # plt.title('Early Stopping Converged Accuracy against Batch Size')
    # plt.plot(batch_sizes, final_acc)
    # plt.xlabel('Batch Size')
    # plt.ylabel('Test Accuracy')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/2c_es_converged_accuracy_against_batch_size.png')

    # # _transform_Y(predicted_dict[64])
    # for batch_size in batch_sizes:
    #     print('Early Stopping Batch size {} Test set classification report:\n{}'.format(batch_size, classification_report(_transform_Y(testY), predicted_dict[batch_size], digits=3, labels=np.unique(predicted_dict[batch_size]))))

    # # Early Stopping Batch size 4 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.980     0.978     0.979       461
    # #           2      0.955     0.938     0.946       224
    # #           3      0.833     0.965     0.894       397
    # #           4      0.553     0.270     0.363       211
    # #           5      0.856     0.781     0.817       237
    # #           7      0.754     0.868     0.807       470
       
    # # avg / total      0.835     0.847     0.834      2000
       
    # # Early Stopping Batch size 8 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.956     0.987     0.971       461
    # #           2      0.972     0.920     0.945       224
    # #           3      0.899     0.877     0.888       397
    # #           4      0.429     0.227     0.297       211
    # #           5      0.824     0.730     0.774       237
    # #           7      0.718     0.921     0.807       470
       
    # # avg / total      0.819     0.832     0.819      2000
       
    # # Early Stopping Batch size 16 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.956     0.989     0.972       461
    # #           2      0.950     0.929     0.939       224
    # #           3      0.886     0.922     0.904       397
    # #           4      0.469     0.322     0.382       211
    # #           5      0.806     0.684     0.740       237
    # #           7      0.741     0.860     0.796       470
       
    # # avg / total      0.822     0.832     0.824      2000

    # # Early Stopping Batch size 32 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.950     0.987     0.968       461
    # #           2      0.932     0.924     0.928       224
    # #           3      0.846     0.955     0.897       397
    # #           4      0.519     0.194     0.283       211
    # #           5      0.788     0.658     0.717       237
    # #           7      0.728     0.889     0.801       470
       
    # # avg / total      0.811     0.828     0.808      2000
       
    # # Early Stopping Batch size 64 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.944     0.987     0.965       461
    # #           2      0.936     0.920     0.928       224
    # #           3      0.841     0.960     0.896       397
    # #           4      0.519     0.190     0.278       211
    # #           5      0.789     0.646     0.710       237
    # #           7      0.730     0.891     0.803       470
       
    # # avg / total      0.809     0.827     0.806      2000


    optimal_batch_size = 16

    # # =====================Q3 Determine optimal number of hidden neurons=====================
    # num_hidden_neurons = [5,10,15,20,25]

    # #### Without Early Stopping
    # train_err_list = []
    # test_acc_list = []
    # time_taken_one_epoch_list = []
    # predicted_dict = dict()

    # for num_neurons in num_hidden_neurons:
    #     classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
    #                             hidden_layer_dict={1: num_neurons}, batch_size=optimal_batch_size).train(small=False, **train_test)
    #     train_err, test_acc, time_taken_one_epoch = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch
    #     train_err_list.append(train_err)
    #     test_acc_list.append(test_acc)
    #     time_taken_one_epoch_list.append(time_taken_one_epoch)
    #     print('{} num_neurons took {}ms per epoch'.format(num_neurons, time_taken_one_epoch))
    #     predicted_dict[num_neurons] = classifier.predict(testX)


    # # Plot Training Errors
    # plt.figure("Train Error against Epoch with different Number of Neurons")
    # plt.title("Train Error against Epoch with different Number of Neurons")
    # plt.axis(error_against_epoch)
    # for i in range(len(num_hidden_neurons)):
    #     plt.plot(range(epochs), train_err_list[i], label = 'num_neurons = {}'.format(num_hidden_neurons[i]))
    #     plt.xlabel(str(epochs) + ' Epochs')
    #     plt.ylabel('Train Error')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/3a_train_error_vs_epoch_for_diff_num_neurons.png')

    # # Plot Test Accuracy
    # plt.figure("Test Accuracy against Epoch with different Number of Neurons")
    # plt.title("Test Accuracy against Epoch with different Number of Neurons")
    # plt.axis(accuracy_against_epoch)
    # for i in range(len(num_hidden_neurons)):
    #     plt.plot(range(epochs), test_acc_list[i], label = 'num_neurons = {}'.format(num_hidden_neurons[i]))
    #     plt.xlabel(str(epochs) + ' Epochs')
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
    # plt.figure('Converged Accuracy against Number of Neurons')
    # plt.title('Converged Accuracy against Number of Neurons')
    # plt.plot(num_hidden_neurons, final_acc)
    # plt.xlabel('Number of Neurons')
    # plt.ylabel('Test Accuracy')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/3c_accuracy_against_num_neurons.png')
    
    # for num_neurons in num_hidden_neurons:
    #     print('Number of Neurons {} Test set classification report:\n{}'.format(num_neurons, classification_report(_transform_Y(testY), predicted_dict[num_neurons], digits=3, labels=np.unique(predicted_dict[num_neurons]))))



    # #### With Early Stopping
    # train_err_list = []
    # test_acc_list = []
    # time_taken_one_epoch_list = []
    # predicted_dict = dict()

    # for num_neurons in num_hidden_neurons:
    #     classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
    #                             hidden_layer_dict={1: num_neurons}, batch_size=optimal_batch_size,
    #                             early_stop=True, patience=20, min_delta=0.005).train(small=False, **train_test)
    #     train_err, test_acc, time_taken_one_epoch = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch
    #     train_err_list.append(train_err)
    #     test_acc_list.append(test_acc)
    #     time_taken_one_epoch_list.append(time_taken_one_epoch)
    #     print('{} num_neurons took {}ms per epoch'.format(num_neurons, time_taken_one_epoch))
    #     predicted_dict[num_neurons] = classifier.predict(testX)
    # #end for

    # # Plot Training Errors
    # plt.figure("Early Stopping Train Error against Epoch with different Number of Neurons")
    # plt.title("Early Stopping Train Error against Epoch with different Number of Neurons")
    # es_error_against_epoch[1] = max([len(l) for l in train_err_list])
    # plt.axis(es_error_against_epoch)

    # for i in range(len(num_hidden_neurons)):
    #     plt.plot(range(len(train_err_list[i])), train_err_list[i], label = 'num_neurons = {}'.format(num_hidden_neurons[i]))
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Train Error')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/3a_es_train_error_vs_epoch_for_diff_num_neurons.png')
    # #end for

    # # Plot Test Accuracy
    # plt.figure("Early Stopping Test Accuracy against Epoch with different Number of Neurons")
    # plt.title("Early Stopping Test Accuracy against Epoch with different Number of Neurons")
    # es_accuracy_against_epoch[1] = max([len(l) for l in test_acc_list])
    # plt.axis(es_accuracy_against_epoch)

    # for i in range(len(num_hidden_neurons)):
    #     plt.plot(range(len(test_acc_list[i])), test_acc_list[i], label = 'num_neurons = {}'.format(num_hidden_neurons[i]))
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Test Accuracy')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/3a_es_test_accuracy_vs_epoch_for_diff_num_neurons.png')
    # #end for

    # # Plot Time Taken for One Epoch
    # plt.figure("Early Stopping Time Taken for One Epoch againt Number of Neurons")
    # plt.title("Early Stopping Time Taken for One Epoch againt Number of Neurons")
    # plt.plot(num_hidden_neurons, time_taken_one_epoch_list)
    # plt.xlabel('Number of Neurons')
    # plt.ylabel('Time/ms')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/3b_es_time_taken_for_one_epoch_vs_num_neurons.png')

    # # plot final test accuracy against Number of Neurons
    # final_acc = [acc[-1] for acc in test_acc_list]
    # plt.figure('Early Stopping Converged Accuracy against Number of Neurons')
    # plt.title('Early Stopping Converged Accuracy against Number of Neurons')
    # plt.plot(num_hidden_neurons, final_acc)
    # plt.xlabel('Number of Neurons')
    # plt.ylabel('Test Accuracy')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/3c_es_accuracy_against_num_neurons.png')
    
    # for num_neurons in num_hidden_neurons:
    #     print('Early Stopping Number of Neurons {} Test set classification report:\n{}'.format(num_neurons, classification_report(_transform_Y(testY), predicted_dict[num_neurons], digits=3, labels=np.unique(predicted_dict[num_neurons]))))

    # # Early Stopping Number of Neurons 5 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.954     0.980     0.967       461
    # #           2      0.924     0.920     0.922       224
    # #           3      0.822     0.967     0.889       397
    # #           4      0.500     0.199     0.285       211
    # #           5      0.765     0.700     0.731       237
    # #           7      0.744     0.847     0.792       470
       
    # # avg / total      0.805     0.824     0.805      2000
       
    # # Early Stopping Number of Neurons 10 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.956     0.989     0.972       461
    # #           2      0.950     0.929     0.939       224
    # #           3      0.886     0.922     0.904       397
    # #           4      0.469     0.322     0.382       211
    # #           5      0.806     0.684     0.740       237
    # #           7      0.741     0.860     0.796       470
       
    # # avg / total      0.822     0.832     0.824      2000
       
    # # Early Stopping Number of Neurons 15 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.962     0.985     0.973       461
    # #           2      0.932     0.920     0.926       224
    # #           3      0.856     0.927     0.890       397
    # #           4      0.488     0.294     0.367       211
    # #           5      0.796     0.675     0.731       237
    # #           7      0.740     0.864     0.797       470
       
    # # avg / total      0.816     0.828     0.817      2000
       
    # # Early Stopping Number of Neurons 20 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.962     0.989     0.975       461
    # #           2      0.924     0.924     0.924       224
    # #           3      0.868     0.929     0.898       397
    # #           4      0.483     0.341     0.400       211
    # #           5      0.797     0.679     0.733       237
    # #           7      0.751     0.840     0.793       470
       
    # # avg / total      0.819     0.830     0.822      2000

    # # Early Stopping Number of Neurons 25 Test set classification report:
    # #              precision    recall  f1-score   support
       
    # #           1      0.956     0.989     0.972       461
    # #           2      0.945     0.920     0.932       224
    # #           3      0.880     0.922     0.900       397
    # #           4      0.471     0.341     0.396       211
    # #           5      0.806     0.684     0.740       237
    # #           7      0.742     0.845     0.790       470
       
    # # avg / total      0.820     0.830     0.822      2000

    optimal_num_neurons = 10

    # =====================Q4 Determine optimal decay parameter=====================
    beta_list = [0,1e-12,1e-9,1e-6,1e-3]

    #### Without Early Stopping
    train_err_list = []
    test_acc_list = []
    time_taken_one_epoch_list = []
    predicted_dict = dict()

    for beta in beta_list:
        classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
                                hidden_layer_dict={1: optimal_num_neurons}, batch_size=optimal_batch_size,
                                l2_beta=beta).train(small=False, **train_test)

        train_err, test_acc, time_taken_one_epoch = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch
        train_err_list.append(train_err)
        test_acc_list.append(test_acc)
        time_taken_one_epoch_list.append(time_taken_one_epoch)
        print('{} beta took {}ms per epoch'.format(beta, time_taken_one_epoch))
        predicted_dict[beta] = classifier.predict(testX)
    #end for

    # Plot Training Errors
    plt.figure("Train Error against Epoch with Different Decay Parameters")
    plt.title("Train Error against Epoch with Different Decay Parameters")
    plt.axis(error_against_epoch)

    for i in range(len(beta_list)):
        plt.plot(range(epochs), train_err_list[i], label = 'beta = {}'.format(beta_list[i]))
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Train Error')
        plt.legend()
        plt.grid(b=True)
        plt.savefig('figures/1a/4a_train_error_vs_epoch_for_diff_beta.png')
    #end for

    # Plot Test Accuracy
    plt.figure("Test Accuracy against Epoch with Different Decay Parameters")
    plt.title("Test Accuracy against Epoch with Different Decay Parameters")
    plt.axis(accuracy_against_epoch)
    for i in range(len(beta_list)):
        plt.plot(range(epochs), test_acc_list[i], label = 'num_neurons = {}'.format(num_hidden_neurons[i]))
        plt.xlabel(str(epochs) + ' Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
        plt.savefig('figures/1a/4b_test_accuracy_vs_epoch_for_diff_beta.png')

    # Plot Time Taken for One Epoch
    plt.figure("Time Taken for One Epoch againt Decay Parameters")
    plt.title("Time Taken for One Epoch againt BeDecay Parametersta")
    plt.plot(beta_list, time_taken_one_epoch_list)
    plt.xlabel('Decay Parameters')
    plt.ylabel('Time/ms')
    plt.grid(b=True)
    plt.savefig('figures/1a/4b_time_taken_for_one_epoch_vs_num_neurons.png')

    # Plot Test Accuracy against Decay Parameters
    final_acc = [acc[-1] for acc in test_acc_list]
    plt.figure('Converged Test Accuracy against Decay Parameters')
    plt.title('Converged Test Accuracy against Decay Parameters')
    plt.xticks(np.arange(5), [str(beta) for beta in beta_list])
    plt.plot([str(beta) for beta in beta_list], final_acc)
    plt.xlabel('Decay Parameters')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/1a/4b_test_accuracy_against_decay_parameters.png')

    #### With Early Stopping
    train_err_list = []
    test_acc_list = []
    time_taken_one_epoch_list = []
    predicted_dict = dict()

    for beta in beta_list:
        classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
                                hidden_layer_dict={1: optimal_num_neurons}, batch_size=optimal_batch_size,
                                l2_beta=beta, early_stop=True, patience=20, min_delta=0.005).train(small=False, **train_test)

        train_err, test_acc, time_taken_one_epoch = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch
        train_err_list.append(train_err)
        test_acc_list.append(test_acc)
        time_taken_one_epoch_list.append(time_taken_one_epoch)
        print('{} beta took {}ms per epoch'.format(beta, time_taken_one_epoch))
        predicted_dict[beta] = classifier.predict(testX)

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
        plt.savefig('figures/1a/4a_es_train_error_vs_epoch_for_diff_beta.png')
   
    # Plot Test Accuracy
    plt.figure("Early Stopping Test Accuracy against Epoch with Different Decay Parameters")
    plt.title("Early Stopping Test Accuracy against Epoch with Different Decay Parameters")
    es_accuracy_against_epoch[1] = max([len(l) for l in test_acc_list])
    plt.axis(es_accuracy_against_epoch)
    for i in range(len(beta_list)):
        plt.plot(range(len(test_acc_list[i])), test_acc_list[i], label = 'num_neurons = {}'.format(num_hidden_neurons[i]))
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
        plt.savefig('figures/1a/4b_test_accuracy_vs_epoch_for_diff_beta.png')

    # Plot Time Taken for One Epoch
    plt.figure("Early Stopping Time Taken for One Epoch againt Decay Parameters")
    plt.title("Early Stopping Time Taken for One Epoch againt Decay Parametersa")
    plt.plot(beta_list, time_taken_one_epoch_list)
    plt.xlabel('Decay Parameters')
    plt.ylabel('Time/ms')
    plt.grid(b=True)
    plt.savefig('figures/1a/4b_time_taken_for_one_epoch_vs_num_neurons.png')

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

    optimal_beta = 0

    # # =====================Q5=====================

    # #### Without Early Stopping
    # train_err_list = []
    # test_acc_list = []
    # time_taken_one_epoch_list =[]
 
    # #### 3-layer Without Early Stopping
    # classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
    #                         hidden_layer_dict={1: 10}).train(small=False, **train_test)

    # train_err1, test_acc1, time_taken_one_epoch1 = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch

    # train_err_list.append(train_err1)
    # test_acc_list.append(test_acc1)
    # time_taken_one_epoch_list.append(time_taken_one_epoch1)
    # print('3-layer took {}ms per epoch'.format(time_taken_one_epoch1))


    # #### 4-layer Without Early Stopping
    # classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
    #                         hidden_layer_dict={1: 10, 2: 10}, num_hidden_layers=2).train(small=False, **train_test)

    # train_err2, test_acc2, time_taken_one_epoch2 = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch

    # train_err_list.append(train_err2)
    # test_acc_list.append(test_acc2)
    # time_taken_one_epoch_list.append(time_taken_one_epoch2)
    # print('4-layer took {}ms per epoch'.format(time_taken_one_epoch2))

    # # plot train_err against epoch
    # plt.figure('Training Error: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.title('Training Error: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.plot(range(epochs), train_err2)
    # plt.xlabel(str(epochs) + ' iterations')
    # plt.ylabel('Train Error')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/5a_train_error_with_4_layer_network.png')

    # # plot test_acc against epoch
    # plt.figure('Test Accuracy: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.title('Test Accuracy: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.plot(range(epochs), test_acc2)
    # plt.xlabel(str(epochs) + ' iterations')
    # plt.ylabel('Test Accuracy')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/5a_test_accuracy_with_4_layer_network.png')

    # # Q5(2) Without Early Stopping
    # # Plot Training Errors
    # plt.figure("Train Error against Epoch with Different Number of Hidden Layers")
    # plt.title("Train Error against Epoch with Different Number of Hidden Layers")
    # plt.axis(error_against_epoch)
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
    # plt.axis(accuracy_against_epoch)

    # for i in range(len(test_acc_list)):
    #     plt.plot(range(epochs), test_acc_list[i], label = 'Number of Hidden Layers = {}'.format(i+1))
    #     plt.xlabel(str(epochs) + ' iterations')
    #     plt.ylabel('Test Accuracy')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/5b_test_accuracy_vs_epoch_for_diff_num_hidden_layers.png')

    # print ("Time taken for 3_layer: %g \nTime taken for 4_layer: %g" % (time_taken_one_epoch_list[0],time_taken_one_epoch_list[1]))


    # #### With Early Stopping
    # train_err_list = []
    # test_acc_list = []
    # time_taken_one_epoch_list =[]
 
    # #### 3-layer With Early Stopping
    # classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
    #                         hidden_layer_dict={1: 10}, early_stop=True, patience=20, min_delta=0.005
    #                         ).train(small=False, **train_test)

    # train_err1, test_acc1, time_taken_one_epoch1 = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch

    # train_err_list.append(train_err1)
    # test_acc_list.append(test_acc1)
    # time_taken_one_epoch_list.append(time_taken_one_epoch1)
    # print('3-layer took {}ms per epoch'.format(time_taken_one_epoch1))

    # #### 4-layer With Early Stopping
    # classifier = Classifier(features_dim=NUM_FEATURES, output_dim=NUM_CLASSES,
    #                         hidden_layer_dict={1: 10, 2: 10}, num_hidden_layers=2,
    #                         early_stop=True, patience=20, min_delta=0.005).train(small=False, **train_test)

    # train_err2, test_acc2, time_taken_one_epoch2 = classifier.train_err, classifier.test_acc, classifier.time_taken_one_epoch
    # print('4-layer took {}ms per epoch'.format(time_taken_one_epoch2))

    # train_err_list.append(train_err2)
    # test_acc_list.append(test_acc2)
    # time_taken_one_epoch_list.append(time_taken_one_epoch2)

    # # plot train_err against epoch
    # plt.figure('Early Stopping Training Error: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.title('Early Stopping Training Error: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.plot(range(len(train_err2)), train_err2)
    # plt.xlabel('Epochs')
    # plt.ylabel('Train Error')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/5a_es_train_error_with_4_layer_network.png')

    # # plot test_acc against epoch
    # plt.figure('Early Stopping Test Accuracy: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.title('Early Stopping Test Accuracy: 2 hidden-layer/batch size 32/10 hidden perceptrons/beta 10^-6')
    # plt.plot(range(len(test_acc2)), test_acc2)
    # plt.xlabel('Epochs')
    # plt.ylabel('Test Accuracy')
    # plt.grid(b=True)
    # plt.savefig('figures/1a/5a_es_test_accuracy_with_4_layer_network.png')

    # # Q5(2) Without Stopping
    # # Plot Training Errors
    # plt.figure("Early Stopping Train Error against Epoch with Different Number of Hidden Layers")
    # plt.title("Early Stopping Train Error against Epoch with Different Number of Hidden Layers")
    # es_error_against_epoch[1] = max([len(l) for l in train_err_list])
    # plt.axis(es_error_against_epoch)

    # for i in range(len(train_err_list)):
    #     plt.plot(range(len(train_err_list[i])), train_err_list[i], label = 'Number of Hidden Layers = {}'.format(i+1))
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Train Error')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/5b_es_train_error_vs_epoch_for_diff_num_hidden_layers.png')

    # # Plot Test Accuracy
    # plt.figure("Early Stopping Test Accuracy against Epoch with Different Number of Hidden Layers")
    # plt.title("Early Stopping Test Accuracy against Epoch with Different Number of Hidden Layers")
    # es_accuracy_against_epoch[1] = max([len(l) for l in test_acc_list])
    # plt.axis(es_accuracy_against_epoch)
    
    # for i in range(len(test_acc_list)):
    #     plt.plot(range(len(test_acc_list[i])), test_acc_list[i], label = 'Number of Hidden Layers = {}'.format(i+1))
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Test Accuracy')
    #     plt.legend()
    #     plt.grid(b=True)
    #     plt.savefig('figures/1a/5b_es_test_accuracy_vs_epoch_for_diff_num_hidden_layers.png')

    # print ("Early Stopping Time taken for 3_layer: %g \nTime taken for 4_layer: %g" % (time_taken_one_epoch_list[0],time_taken_one_epoch_list[1]))

#end def

if __name__ == '__main__': main()
