#
# Project 1, starter code part a
#

from collections import OrderedDict
import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import time
import os

if not os.path.exists('figures'):
    os.makedirs('figures')
if not os.path.exists(os.path.join('figures', 'a')):
    os.makedirs(os.path.join('figures', 'a'))

if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists(os.path.join('models', 'a')):
    os.makedirs(os.path.join('models', 'a'))

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
seed = 10
learning_rate = 0.001
np.random.seed(seed)
tf.set_random_seed(seed)

class CNNClassifer():
    def __init__(
        self, save_path,
        input_width = IMG_SIZE, input_height = IMG_SIZE, num_channels = NUM_CHANNELS, output_dim = NUM_CLASSES, drop_out=False,
        keep_prob=0.9, hidden_layer_dict=None,
        batch_size=128, learning_rate= 0.001, epochs=1000,
        early_stop= True, patience=20, min_delta=0.001,
        optimizer='GD',momentum=0.1,
        **kwargs
    ):
        self.save_path = save_path
        self.input_width = input_width
        self.input_height = input_height
        self.num_channels = num_channels
        self.output_dim = output_dim
        self.drop_out = drop_out
        if optimizer == 'Drop Out':
            self.drop_out = True
        if self.drop_out:
            self._keep_prob = tf.placeholder(tf.float32)
            self.keep_prob = keep_prob
        self.hidden_layer_dict = hidden_layer_dict
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        self.optimizer = optimizer
        self.momentum = momentum
        self._build_model()

    #end def

    def _build_layer(self, x, cwindow_width, cwindow_height, input_maps, output_maps, cstrides, cpadding,
                    pwindow_width, pwindow_height, pstrides, ppadding, **kwargs):
        #Conv
        W = tf.Variable(tf.truncated_normal([cwindow_width, cwindow_height, input_maps, output_maps], seed=seed,
                        stddev=1.0/np.sqrt(input_maps*cwindow_width*cwindow_height)), name='weights')  # [window_width, window_height, input_maps, output_maps]
        b = tf.Variable(tf.zeros([output_maps]), name='biases')

        conv = tf.nn.relu(tf.nn.conv2d(x, W, [1, cstrides, cstrides, 1], padding=cpadding) + b)  # strides = [1, stride, stride, 1]

        if self.drop_out:
            conv = tf.nn.dropout(conv, self._keep_prob)

        #Pool
        pool = tf.nn.max_pool(conv, ksize=[1, pwindow_width, pwindow_height, 1], strides= [1, pstrides, pstrides, 1], padding=ppadding, name='pool')

        return W, conv, pool
    #end def


    def _build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.input_width*self.input_height*self.num_channels])
        self.y_ = tf.placeholder(tf.float32, [None, self.output_dim])

        self.x_ = tf.reshape(self.x, [-1, self.input_width, self.input_height, self.num_channels])  # [no_patterns, input_width, input_height, no_input_maps]

        # Conv 1 and pool 1
        input_dict1 = dict(cwindow_width=self.hidden_layer_dict['C1']['window_width'], cwindow_height=self.hidden_layer_dict['C1']['window_height'],
                        input_maps=self.num_channels, output_maps=self.hidden_layer_dict['C1']['output_maps'], cstrides=self.hidden_layer_dict['C1']['strides'],
                        cpadding=self.hidden_layer_dict['C1']['padding'],
                        pwindow_width=self.hidden_layer_dict['S1']['window_width'], pwindow_height=self.hidden_layer_dict['S1']['window_height'],
                        pstrides=self.hidden_layer_dict['S1']['strides'], ppadding=self.hidden_layer_dict['S1']['padding'])
        self.W_conv1, self.h_conv1, self.h_pool1 = self._build_layer(self.x_, **input_dict1)

        # Conv 2 and pool 2
        input_dict2 = dict(cwindow_width=self.hidden_layer_dict['C2']['window_width'], cwindow_height=self.hidden_layer_dict['C2']['window_height'],
                        input_maps=self.hidden_layer_dict['C1']['output_maps'], output_maps=self.hidden_layer_dict['C2']['output_maps'], cstrides=self.hidden_layer_dict['C2']['strides'],
                        cpadding=self.hidden_layer_dict['C2']['padding'],
                        pwindow_width=self.hidden_layer_dict['S2']['window_width'], pwindow_height=self.hidden_layer_dict['S2']['window_height'],
                        pstrides=self.hidden_layer_dict['S2']['strides'], ppadding=self.hidden_layer_dict['S2']['padding'])
        self.W_conv2, self.h_conv2, self.h_pool2 = self._build_layer(self.h_pool1, **input_dict2)

        # Fully connected layer F3 of size 300
        dim = self.h_pool2.get_shape()[1].value * self.h_pool2.get_shape()[2].value * self.h_pool2.get_shape()[3].value
        h_pool2_flat = tf.reshape(self.h_pool2, [-1, dim])
        self.Wf = tf.Variable(tf.truncated_normal([dim, self.hidden_layer_dict['F1']['size']], seed=seed, stddev=1.0/np.sqrt(dim)), name='weights')
        self.bf = tf.Variable(tf.zeros([self.hidden_layer_dict['F1']['size']]), name='biases')
        self.hf = tf.nn.relu(tf.matmul(h_pool2_flat, self.Wf) + self.bf)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        if self.drop_out:
            self.hf = tf.nn.dropout(self.hf, self._keep_prob)

        self.W_output = tf.Variable(tf.truncated_normal([self.hidden_layer_dict['F1']['size'], self.output_dim], seed=seed, stddev=1.0/np.sqrt(self.hidden_layer_dict['F1']['size'])), name='weights')
        self.b_output = tf.Variable(tf.zeros([self.output_dim]), name='biases')
        self.y_conv = tf.matmul(self.hf, self.W_output) + self.b_output

        # accuracy
        self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1)), tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))

        # diff choices of optimizer
        if self.optimizer == 'momentum': self.train_op = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cross_entropy)
        elif self.optimizer == 'RMSProp': self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cross_entropy)
        elif self.optimizer == 'Adam': self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
        else: self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
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

                if self.drop_out:
                    self.train_err.append(self.cross_entropy.eval(feed_dict={self.x: X_train, self.y_: Y_train, self._keep_prob: self.keep_prob}))
                    self.test_acc.append(self.accuracy.eval(feed_dict={self.x: X_test, self.y_: Y_test, self._keep_prob: 1.0}))
                else:
                    self.train_err.append(self.cross_entropy.eval(feed_dict={self.x: X_train, self.y_: Y_train}))
                    self.test_acc.append(self.accuracy.eval(feed_dict={self.x: X_test, self.y_: Y_test}))

                if self.early_stop:
                    if self.drop_out:
                        _val_err = self.cross_entropy.eval(feed_dict={self.x: X_val, self.y_: Y_val, self._keep_prob: self.keep_prob})
                    else:
                        _val_err = self.cross_entropy.eval(feed_dict={self.x: X_val, self.y_: Y_val})
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

                # if i % 100 == 0:
                #     print('iter: %d, train error  : %g'%(i, self.train_err[i]))
                #     print('iter: %d, test accuracy  : %g'%(i, self.test_acc[i]))
                #     print('-'*50)

            #end for
            self.early_stop_epoch = _epochs
            self.saver.save(sess, self.save_path)
        #end with

        self.time_taken_one_epoch = (time_to_update/_epochs) * 1000
        return self
    #end def


    def get_feature_maps(self, X):
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_path)
            c1, p1, c2, p2 = sess.run([self.h_conv1, self.h_pool1, self.h_conv2, self.h_pool2], {self.x: X.reshape(-1, self.input_width*self.input_height*self.num_channels)})
        return c1, p1, c2, p2
    #end def
#end class


def load_data(file):
    #load file
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels-1] = 1

    return data, labels_

# data scaling
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)
#end def


# read data, preprocess data into train, validation and test, with min_max scaling
def read_data(file_train, file_test):
    # load data
    trainX, trainY = load_data(file_train)
    print(trainX.shape, trainY.shape)
    testX, testY = load_data(file_test)
    print(testX.shape, testY.shape)
    # split train set and validation set by 3:1
    X_train, X_val, Y_train, Y_val = train_test_split(trainX, trainY, test_size=0.20, random_state=seed)
    X_min = np.min(trainX, axis = 0)
    X_max = np.max(trainX, axis = 0)

    # scaling
    X_train = scale(X_train, X_min, X_max)
    X_val = scale(X_val, X_min, X_max)
    testX = scale(testX, X_min, X_max)

    return X_train, Y_train, X_val, Y_val, testX, testY
#end def


def plot_feature_map(X, cnn, path_dict):
    # original test pattern
    plt.figure()
    plt.gray()
    X_show = X.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
    plt.axis('off')
    plt.imshow(X_show)
    plt.savefig(path_dict['test'])

    # feed X into cnn
    c1, p1, c2, p2 = cnn.get_feature_maps(X)

    # plot c1
    plt.figure('C1')
    plt.title('C1')
    plt.gray()
    plt.subplot(3,1,1), plt.axis('off'), plt.imshow(c1[0,:,:,0])
    plt.subplot(3,1,2), plt.axis('off'), plt.imshow(c1[0,:,:,1])
    plt.subplot(3,1,3), plt.axis('off'), plt.imshow(c1[0,:,:,2])
    plt.savefig(path_dict['c1'])

    # plot p1
    plt.figure('P1')
    plt.title('P1')
    plt.gray()
    plt.subplot(3,1,1), plt.axis('off'), plt.imshow(p1[0,:,:,0])
    plt.subplot(3,1,2), plt.axis('off'), plt.imshow(p1[0,:,:,1])
    plt.subplot(3,1,3), plt.axis('off'), plt.imshow(p1[0,:,:,2])
    plt.savefig(path_dict['p1'])

    # plot c2
    plt.figure('C2')
    plt.title('C2')
    plt.gray()
    plt.subplot(3,1,1), plt.axis('off'), plt.imshow(c2[0,:,:,0])
    plt.subplot(3,1,2), plt.axis('off'), plt.imshow(c2[0,:,:,1])
    plt.subplot(3,1,3), plt.axis('off'), plt.imshow(c2[0,:,:,2])
    plt.savefig(path_dict['c2'])

    # plot p2
    plt.figure('P2')
    plt.title('P2')
    plt.gray()
    plt.subplot(3,1,1), plt.axis('off'), plt.imshow(p2[0,:,:,0])
    plt.subplot(3,1,2), plt.axis('off'), plt.imshow(p2[0,:,:,1])
    plt.subplot(3,1,3), plt.axis('off'), plt.imshow(p2[0,:,:,2])
    plt.savefig(path_dict['p2'])
#end def


def arg_dict(model_save_path, C1_map=50, C2_map=60,optimizer='GD'):
    C1_dict = dict(window_width=9, window_height=9, output_maps=C1_map, padding='VALID', strides=1)
    C2_dict = dict(window_width=5, window_height=5, output_maps=C2_map, padding='VALID', strides=1)
    S1_dict = dict(window_width=2, window_height=2, padding='VALID', strides=2)
    S2_dict = dict(window_width=2, window_height=2, padding='VALID', strides=2)
    F1_dict = dict(size=300)
    hidden_layer_dict = dict(C1=C1_dict, C2=C2_dict, S1=S1_dict, S2=S2_dict, F1=F1_dict)

    init_dict = dict(save_path=model_save_path,optimizer=optimizer,
        input_width=IMG_SIZE, input_height=IMG_SIZE, num_channels=NUM_CHANNELS, output_dim=NUM_CLASSES,
        hidden_layer_dict=hidden_layer_dict,
        batch_size=128, learning_rate=learning_rate, epochs=1000,
        early_stop=True, patience=20, min_delta=0.001)
    return init_dict
#end def


def grid_search(trainX, trainY,
                testX, testY,
                valX, valY,
                C1_map_range=[10,100,20], C2_map_range=[10,100,20]):
    optimizer = 'GD'
    gs_train_err_dict = dict()
    gs_test_acc_dict = dict()
    gs_time_taken_one_epoch_dict = dict()
    gs_early_stop_epoch_dict = dict()
    for C1_map in range(C1_map_range[0], C1_map_range[1], C1_map_range[2]):
        if C1_map <= 0: continue
        for C2_map in range(C2_map_range[0], C2_map_range[1], C2_map_range[2]):
            if C2_map <= 0: continue
            key = (C1_map, C2_map)
            model_save_path = 'models/a/2_C1_' + str(C1_map) + '_C2_' + str(C2_map)
            init_dict = arg_dict(model_save_path, C1_map, C2_map,optimizer)
            cnn = CNNClassifer(**init_dict).train(X_train=trainX, Y_train=trainY,
                                                X_test=testX, Y_test=testY,
                                                X_val=valX, Y_val=valY)
            train_err, test_acc, time_taken_one_epoch, early_stop_epoch = cnn.train_err, cnn.test_acc, cnn.time_taken_one_epoch, cnn.early_stop_epoch

            gs_train_err_dict[key] = train_err[-1]
            gs_test_acc_dict[key] = test_acc[-1]
            gs_time_taken_one_epoch_dict[key] = time_taken_one_epoch
            gs_early_stop_epoch_dict[key] = early_stop_epoch

    gs_test_acc_dict = OrderedDict(sorted(gs_test_acc_dict.items(), key=lambda t: t[1], reverse=True))
    print(gs_test_acc_dict)
    opt_C1_C2 = list(gs_test_acc_dict.items())[0]
    optimal_C1 = opt_C1_C2[0][0]
    optimal_C2 = opt_C1_C2[0][1]
    optimal_test_acc = opt_C1_C2[1]
    optimal_train_err = gs_train_err_dict[opt_C1_C2[0]]
    optimal_time_taken_one_epoch = gs_time_taken_one_epoch_dict[opt_C1_C2[0]]
    optimal_early_stop_epoch = gs_early_stop_epoch_dict[opt_C1_C2[0]]

    print('C1: %d\nC2: %d\ntest_acc: %f' % (optimal_C1,optimal_C2,optimal_test_acc))

    optimal_dict = dict(C1=optimal_C1, C2=optimal_C2, test_acc=optimal_test_acc, train_err=optimal_train_err, time=optimal_time_taken_one_epoch, es_epoch=optimal_early_stop_epoch)
    return optimal_dict
#end def


def main():
    error_against_epoch = [0, 1000, 0, 2]
    accuracy_against_epoch = [0, 1000, 0, 1]

    train_err_dict = dict()
    test_acc_dict = dict()
    time_taken_one_epoch_dict = dict()
    early_stop_epoch_dict = dict()

    trainX, trainY, valX, valY, testX, testY = read_data('data/data_batch_1', 'data/test_batch_trim')

    # trainX = trainX[:200]
    # trainY = trainY[:200]
    # valX = valX[:10]
    # valY = valY[:10]
    # testX = testX[:10]
    # testY = testY[:10]

    # =====================Q1 =====================
    model_save_path = 'models/a/1_GD'
    C1_map = 50
    C2_map = 60
    optimizer = 'GD'
    init_dict = arg_dict(model_save_path, C1_map, C2_map,optimizer)
    cnn = CNNClassifer(**init_dict).train(X_train=trainX, Y_train=trainY,
                                        X_test=testX, Y_test=testY,
                                        X_val=valX, Y_val=valY)
    train_err, test_acc, time_taken_one_epoch, early_stop_epoch = cnn.train_err, cnn.test_acc, cnn.time_taken_one_epoch, cnn.early_stop_epoch
    train_err_dict['GD_50_60'] = train_err
    test_acc_dict['GD_50_60'] = test_acc
    time_taken_one_epoch_dict['GD_50_60'] = time_taken_one_epoch
    early_stop_epoch_dict['GD_50_60'] = early_stop_epoch

    # Plot Training Errors
    plt.figure("Train Error against Epoch")
    plt.title("Train Error against Epoch")
    plt.plot(range(len(train_err)), train_err)
    plt.xlabel('Epochs')
    plt.ylabel('Train Error')
    plt.grid(b=True)
    plt.savefig('figures/a/1a_train_error_vs_epoch.png')

    # Plot Test Accuracy
    plt.figure("Early Stopping Test Accuracy against Epoch")
    plt.title("Early Stopping Test Accuracy against Epoch")
    plt.plot(range(len(test_acc)), test_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    plt.savefig('figures/a/1a_test_accuracy_vs_epoch.png')

    np.random.seed(seed)

    ind = np.random.randint(low=0, high=len(testX))
    X = trainX[ind,:]
    path_dict = dict(test='figures/a/1b_1_test.png', c1='figures/a/1b_1_c1.png',
                    p1='figures/a/1b_1_p1.png', c2='figures/a/1b_1_c2.png',
                    p2='figures/a/1b_1_p2.png')
    plot_feature_map(X, cnn, path_dict)

    ind = np.random.randint(low=0, high=10)
    X = trainX[ind,:]
    path_dict = dict(test='figures/a/1b_2_test.png', c1='figures/a/1b_2_c1.png',
                    p1='figures/a/1b_2_p1.png', c2='figures/a/1b_2_c2.png',
                    p2='figures/a/1b_2_p2.png')
    plot_feature_map(X, cnn, path_dict)

    # =====================Q2 optimal feature map=====================
    C1_range = [40,71,10]
    C2_range = [40,71,10]
    optimal_dict = grid_search(trainX, trainY,
                                testX, testY,
                                valX, valY,
                                C1_range, C2_range)

    C1 = optimal_dict['C1']
    C1_range = [C1-10,C1+10,5]
    C2 = optimal_dict['C2']
    C2_range = [C1-10,C1+10,5]
    optimal_dict = grid_search(trainX, trainY,
                                testX, testY,
                                valX, valY,
                                C1_range, C2_range)

    # C1 = optimal_dict['C1']
    # C1_range = [C1-10,C1+10,5]
    # C2 = optimal_dict['C2']
    # C2_range = [C1-10,C1+10,5]
    # optimal_dict = grid_search(trainX, trainY,
    #                             testX, testY,
    #                             valX, valY,
    #                             C1_range, C2_range) 

    # C1 = optimal_dict['C1']
    # C1_range = [C1-5,C1+5,2]
    # C2 = optimal_dict['C2']
    # C2_range = [C1-5,C1+5,2]
    # optimal_dict = grid_search(trainX, trainY,
    #                             testX, testY,
    #                             valX, valY,
    #                             C1_range, C2_range)

    # C1 = optimal_dict['C1']
    # C1_range = [C1-2,C1+2,1]
    # C2 = optimal_dict['C2']
    # C2_range = [C1-2,C1+2,1]
    # optimal_dict = grid_search(trainX, trainY,
    #                             testX, testY,
    #                             valX, valY,
    print("Optimal C1: {}\n Optimal C2: {}".format(optimal_dict['C1'], optimal_dict['C2']))

    # # =====================Q3 optimal feature map=====================
    # optimizers = ['momentum','RMSProp','Adam','Drop Out']

    # for optimizer in optimizers:
    #     print('='*50)
    #     print(optimizer.values())
    #     model_save_path = 'models/a/3_' + str(optimizer)
    #     C1_map = 50
    #     C2_map = 60
    #     init_dict = arg_dict(model_save_path, C1_map, C2_map,optimizer)
    #     cnn = CNNClassifer(**init_dict).train(X_train=trainX, Y_train=trainY,
    #                                         X_test=testX, Y_test=testY,
    #                                         X_val=valX, Y_val=valY)
    #     train_err, test_acc, time_taken_one_epoch, early_stop_epoch = cnn.train_err, cnn.test_acc, cnn.time_taken_one_epoch, cnn.early_stop_epoch

    #     train_err_dict[optimizer] = train_err
    #     test_acc_dict[optimizer] = test_acc
    #     time_taken_one_epoch_dict[optimizer] = time_taken_one_epoch
    #     early_stop_epoch_dict[optimizer] = early_stop_epoch
    # #end for

    # # Plot Training Errors
    # plt.figure("Train Error against Epoch")
    # plt.title("Train Error against Epoch")
    # error_against_epoch[1] = max([len(l) for l in train_err_dict.values()])
    # plt.axis(error_against_epoch)
    # for key, val in train_err_dict.items():
    #     plt.plot(range(len(val)), val, label = 'optimizer = {}'.format(key))
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Train Error')
    #     plt.legend()
    #     plt.grid(b=True)
    # #end for
    # plt.savefig('figures/a/3_train_error_vs_epoch.png')

    # # Plot Test Accuracy
    # plt.figure("Test Accuracy against Epoch")
    # plt.title("Test Accuracy against Epoch")
    # accuracy_against_epoch[1] = max([len(l) for l in test_acc_dict.values()])
    # plt.axis(accuracy_against_epoch)
    # for key, val in test_acc_dict.items():
    #     plt.plot(range(len(val)), val, label = 'optimizer = {}'.format(key))
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Test Accuracy')
    #     plt.legend()
    #     plt.grid(b=True)
    # #end for
    # plt.savefig('figures/a/3_test_accuracy_vs_epoch.png')
    # #end for

    # # Plot Time Taken for One Epoch
    # optimizer_list = test_acc_dict.keys()
    # time_taken_one_epoch_list = [time_taken_one_epoch_dict[optimizer] for optimizer in optimizer_list]
    # plt.figure("Time Taken for One Epoch")
    # plt.title("Time Taken for One Epoch")
    # plt.xticks(np.arange(len(optimizer_list)), optimizer_list)
    # plt.plot(optimizer_list, time_taken_one_epoch_list)
    # plt.xlabel('Optimizer')
    # plt.ylabel('Time per Epoch/ms')
    # plt.grid(b=True)
    # plt.savefig('figures/a/3_time_taken_for_one_epoch.png')

    # early_stop_epoch_list = [early_stop_epoch_dict[optimizer] for optimizer in optimizer_list]
    # total_time_taken_list = [x*y for x,y in zip(early_stop_epoch_list,time_taken_one_epoch_list)]
    # # Plot Total Time Taken
    # plt.figure("Early Stopping Total Time Taken")
    # plt.title("Early Stopping Total Time Taken")
    # plt.plot(optimizer_list, total_time_taken_list)
    # plt.xlabel('Optimizer')
    # plt.ylabel('Total Time/ms')
    # plt.grid(b=True)
    # plt.savefig('figures/a/3_total_time_taken.png')

    # # =====================Q4 comparison of model accuracy=====================
    # test_acc_list = []
    # model_list = []

    # for key, val in test_acc_dict.items():
    #     model_list.append(key)
    #     test_acc_list.append(val)
    #     print("model: %s test_acc: %f" %(key,val))

    # # Plot Test Accuracy
    # plt.figure("Test Accuracy against models")
    # plt.title("Test Accuracy against models")
    # plt.grid(b=True)
    # plt.ylabel('Test Accuracy')
    # plt.xticks(np.arange(6), state_list)
    # plt.plot(model_list, test_acc_list)
    # plt.savefig('figures/a/4_model_comparison.png')

# end def

if __name__ == '__main__': main()
