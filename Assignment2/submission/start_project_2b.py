import csv
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import pandas as pd
import pylab as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15

learning_rate = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists(os.path.join('models', 'b')):
    os.makedirs(os.path.join('models', 'b'))
if not os.path.exists(os.path.join('csv_results')):
    os.makedirs(os.path.join('csv_results'))

if not os.path.exists('figures'):
    os.makedirs('figures')
if not os.path.exists(os.path.join('figures', 'b')):
    os.makedirs(os.path.join('figures', 'b'))


class CNNClassifer():
    def __init__(
        self, save_path,
        input_dim=100, output_dim=15, drop_out=False,
        keep_prob=0.5, hidden_layer_dict=None,
        batch_size=128, learning_rate=0.01,
        epochs=1000,
        early_stop=False, patience=20, min_delta=0.001, min_epoch=20,
        choice='char', n_words=None, embedding_size=None,
        **kwargs
    ):
        self.save_path = save_path
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_out = drop_out
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
        self.min_epoch = min_epoch
        if choice == 'char':
            self._build_char_model()
        else:
            self.n_words = n_words
            self.embedding_size = embedding_size
            self._build_word_model()
    #end def


    def _build_layer(self, x, cfilters, ckernel_size, cpadding,
                    pwindow, pstrides, ppadding, **kwargs):
        # Conv
        conv = tf.layers.conv2d(
                            x,
                            filters=cfilters,
                            kernel_size=ckernel_size,
                            padding=cpadding,
                            activation=tf.nn.relu)

        # Pool
        pool = tf.layers.max_pooling2d(
                            conv,
                            pool_size=pwindow,
                            strides=pstrides,
                            padding=ppadding)

        if self.drop_out:
            pool = tf.nn.dropout(pool, self._keep_prob)

        return conv, pool
    #end def


    def _build_char_model(self):
        self.x = tf.placeholder(tf.int64, [None, self.input_dim])
        self.y_ = tf.placeholder(tf.int64)
        self.x_ = tf.reshape(tf.one_hot(self.x, 256), [-1, self.input_dim, 256, 1])

        # Conv 1 and pool 1
        input_dict1 = dict(
                        cfilters=self.hidden_layer_dict['C1']['filters'],
                        ckernel_size=self.hidden_layer_dict['C1']['kernel_size'],
                        cpadding=self.hidden_layer_dict['C1']['padding'],
                        pwindow=self.hidden_layer_dict['S1']['window'],
                        pstrides=self.hidden_layer_dict['S1']['strides'],
                        ppadding=self.hidden_layer_dict['S1']['padding'])
        self.h_conv1, self.h_pool1 = self._build_layer(self.x_, **input_dict1)

        # Conv 2 and pool 2
        input_dict2 = dict(
                        cfilters=self.hidden_layer_dict['C2']['filters'],
                        ckernel_size=self.hidden_layer_dict['C2']['kernel_size'],
                        cpadding=self.hidden_layer_dict['C2']['padding'],
                        pwindow=self.hidden_layer_dict['S2']['window'],
                        pstrides=self.hidden_layer_dict['S2']['strides'],
                        ppadding=self.hidden_layer_dict['S2']['padding'])
        self.h_conv2, self.h_pool2 = self._build_layer(self.h_pool1, **input_dict2)

        self.h_pool2 = tf.squeeze(tf.reduce_max(self.h_pool2, 1), squeeze_dims=[1])
        self.y_conv = tf.layers.dense(self.h_pool2, self.output_dim, activation=None)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.y_, self.output_dim), logits=self.y_conv))

        # accuracy
        self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(tf.one_hot(self.y_, self.output_dim), 1)), tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
    #end def


    def _build_word_model(self):
        self.x = tf.placeholder(tf.int64, [None, self.input_dim])
        self.y_ = tf.placeholder(tf.int64)
        self.x_ = tf.contrib.layers.embed_sequence(
                                                self.x,
                                                vocab_size=self.n_words,
                                                embed_dim=self.embedding_size)
        self.x_ = tf.reshape(self.x_, [-1, self.input_dim, self.embedding_size, 1])

        # Conv 1 and pool 1
        input_dict1 = dict(
                        cfilters=self.hidden_layer_dict['C1']['filters'],
                        ckernel_size=self.hidden_layer_dict['C1']['kernel_size'],
                        cpadding=self.hidden_layer_dict['C1']['padding'],
                        pwindow=self.hidden_layer_dict['S1']['window'],
                        pstrides=self.hidden_layer_dict['S1']['strides'],
                        ppadding=self.hidden_layer_dict['S1']['padding'])
        self.h_conv1, self.h_pool1 = self._build_layer(self.x_, **input_dict1)

        # Conv 2 and pool 2
        input_dict2 = dict(
                        cfilters=self.hidden_layer_dict['C2']['filters'],
                        ckernel_size=self.hidden_layer_dict['C2']['kernel_size'],
                        cpadding=self.hidden_layer_dict['C2']['padding'],
                        pwindow=self.hidden_layer_dict['S2']['window'],
                        pstrides=self.hidden_layer_dict['S2']['strides'],
                        ppadding=self.hidden_layer_dict['S2']['padding'])
        self.h_conv2, self.h_pool2 = self._build_layer(self.h_pool1, **input_dict2)

        self.h_pool2 = tf.squeeze(tf.reduce_max(self.h_pool2, 1), squeeze_dims=[1])
        self.y_conv = tf.layers.dense(self.h_pool2, self.output_dim, activation=None)

        self.cross_entropy = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits_v2(
                                                        labels=tf.one_hot(self.y_, self.output_dim),
                                                        logits=self.y_conv))

        # accuracy
        self.correct_prediction = tf.cast(
                                    tf.equal(
                                        tf.argmax(self.y_conv, 1),
                                        tf.argmax(tf.one_hot(self.y_, self.output_dim), 1)), tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
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
                    if self.drop_out:
                        self.train_op.run(feed_dict={self.x: X_train[_start:_end], self.y_: Y_train[_start:_end], self._keep_prob: self.keep_prob})
                    else:
                        self.train_op.run(feed_dict={self.x: X_train[_start:_end], self.y_: Y_train[_start:_end]})
                time_to_update += (time.time() - t)

                if self.drop_out:
                    self.train_err.append(self.cross_entropy.eval(feed_dict={self.x: X_train, self.y_: Y_train, self._keep_prob: self.keep_prob}))
                    self.test_acc.append(self.accuracy.eval(feed_dict={self.x: X_test, self.y_: Y_test, self._keep_prob: 1.0}))
                else:
                    self.train_err.append(self.cross_entropy.eval(feed_dict={self.x: X_train, self.y_: Y_train}))
                    self.test_acc.append(self.accuracy.eval(feed_dict={self.x: X_test, self.y_: Y_test}))

                if self.early_stop:
                    if self.drop_out: _val_err = self.cross_entropy.eval(feed_dict={self.x: X_val, self.y_: Y_val, self._keep_prob: self.keep_prob})
                    else: _val_err = self.cross_entropy.eval(feed_dict={self.x: X_val, self.y_: Y_val})

                    if _epochs > self.min_epoch:
                        if (tmp_best_val_err - _val_err) < self.min_delta:
                            _patience -= 1
                            if _patience <= 0:
                                print('Early stopping at {}th iteration. Test Acc {}'.format(i, self.test_acc[-1]))
                                break
                        else:
                            _patience = self.patience
                            tmp_best_val_err = _val_err
                        #end if
                    #end if
                #end if

                if i % 100 == 0:
                    print('iter: %d, train error  : %g'%(i, self.train_err[i]))
                    print('iter: %d, test accuracy  : %g'%(i, self.test_acc[i]))

            #end for
            self.early_stop_epoch = _epochs
            self.saver.save(sess, self.save_path)
        #end with

        self.time_taken_one_epoch = (time_to_update/_epochs) * 1000
        return self
    #end def
#end class


class RNNClassifer():
    def __init__(
        self, save_path,
        input_dim=100, output_dim=15, drop_out=False,
        keep_prob=0, n_hidden_list=None,
        batch_size=128, learning_rate=0.01,
        epochs=1000,
        early_stop=False, patience=20, min_delta=0.001, min_epoch=20,
        choice='char', n_words=None, embedding_size=None,
        rnn_choice='GRU',
        gradient_clipped=False,
        **kwargs
    ):
        self.save_path = save_path
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_out = drop_out
        if self.drop_out:
            self._keep_prob = tf.placeholder(tf.float32)
            self.keep_prob = keep_prob
        self.n_hidden_list = n_hidden_list
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        self.min_epoch = min_epoch
        self.rnn_choice = rnn_choice
        self.gradient_clipped = gradient_clipped
        if choice == 'char':
            self._build_char_model()
        else:
            self.n_words = n_words
            self.embedding_size = embedding_size
            self._build_word_model()
    #end def


    def _build_layer(self, x, n_hidden_list, rnn_choice='GRU', **kwargs):
        if rnn_choice == 'GRU':
            cells = [tf.nn.rnn_cell.GRUCell(n) for n in n_hidden_list]
            cells = tf.nn.rnn_cell.MultiRNNCell(cells)
            outputs, states = tf.nn.dynamic_rnn(cells, x, dtype=tf.float32)
        elif rnn_choice == 'BASIC':
            cells = [tf.nn.rnn_cell.BasicRNNCell(n) for n in n_hidden_list]
            cells = tf.nn.rnn_cell.MultiRNNCell(cells)
            outputs, states = tf.nn.dynamic_rnn(cells, x, dtype=tf.float32)
        elif rnn_choice == 'LSTM':
            cells = [tf.nn.rnn_cell.LSTMCell(n) for n in n_hidden_list]
            cells = tf.nn.rnn_cell.MultiRNNCell(cells)
            outputs, states = tf.nn.dynamic_rnn(cells, x, dtype=tf.float32)
            states = states[-1].h

        return states if not isinstance(states, tuple) else states[-1]
    #end def


    def _build_char_model(self):
        self.x = tf.placeholder(tf.int64, [None, self.input_dim])
        self.y_ = tf.placeholder(tf.int64)
        self.x_ = tf.reshape(tf.one_hot(self.x, 256), [-1, self.input_dim, 256])

        input_dict1 = dict(
                        n_hidden_list=self.n_hidden_list,
                        rnn_choice=self.rnn_choice)
        self.encoding = self._build_layer(self.x_, **input_dict1)

        if self.drop_out:
            self.encoding = tf.nn.dropout(self.encoding, self._keep_prob)

        self.y = tf.layers.dense(self.encoding, self.output_dim, activation=None)

        self.cross_entropy = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits_v2(
                                                        labels=tf.one_hot(self.y_, self.output_dim),
                                                        logits=self.y))

        # accuracy
        self.correct_prediction = tf.cast(
                                    tf.equal(
                                        tf.argmax(self.y, 1),
                                        tf.argmax(tf.one_hot(self.y_, self.output_dim), 1)), tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)

        if self.gradient_clipped:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gvs = optimizer.compute_gradients(self.cross_entropy)
            capped_gvs = [(tf.clip_by_value(grad, clip_value_min=-2., clip_value_max=2.), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs)
        else:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
    #end def


    def _build_word_model(self):
        self.x = tf.placeholder(tf.int64, [None, self.input_dim])
        self.y_ = tf.placeholder(tf.int64)
        self.x_ = tf.contrib.layers.embed_sequence(
                                                self.x,
                                                vocab_size=self.n_words,
                                                embed_dim=self.embedding_size)
        self.x_ = tf.reshape(self.x_, [-1, self.input_dim, self.embedding_size])

        input_dict1 = dict(
                        n_hidden_list=self.n_hidden_list,
                        rnn_choice=self.rnn_choice)
        self.encoding = self._build_layer(self.x_, **input_dict1)

        if self.drop_out:
            self.encoding = tf.nn.dropout(self.encoding, self._keep_prob)

        self.y = tf.layers.dense(self.encoding, self.output_dim, activation=None)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.y_, self.output_dim), logits=self.y))

        # accuracy
        self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(tf.one_hot(self.y_, self.output_dim), 1)), tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)
        if self.gradient_clipped:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gvs = optimizer.compute_gradients(self.cross_entropy)
            capped_gvs = [(tf.clip_by_value(grad, clip_value_min=-2., clip_value_max=2.), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs)
        else:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
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
                    if self.drop_out:
                        self.train_op.run(feed_dict={self.x: X_train[_start:_end], self.y_: Y_train[_start:_end], self._keep_prob: self.keep_prob})
                    else:
                        self.train_op.run(feed_dict={self.x: X_train[_start:_end], self.y_: Y_train[_start:_end]})
                time_to_update += (time.time() - t)

                if self.drop_out:
                    self.train_err.append(self.cross_entropy.eval(feed_dict={self.x: X_train, self.y_: Y_train, self._keep_prob: self.keep_prob}))
                    self.test_acc.append(self.accuracy.eval(feed_dict={self.x: X_test, self.y_: Y_test, self._keep_prob: 1.0}))
                else:
                    self.train_err.append(self.cross_entropy.eval(feed_dict={self.x: X_train, self.y_: Y_train}))
                    self.test_acc.append(self.accuracy.eval(feed_dict={self.x: X_test, self.y_: Y_test}))

                if self.early_stop:
                    if self.drop_out: _val_err = self.cross_entropy.eval(feed_dict={self.x: X_val, self.y_: Y_val, self._keep_prob: self.keep_prob})
                    else: _val_err = self.cross_entropy.eval(feed_dict={self.x: X_val, self.y_: Y_val})

                    if (tmp_best_val_err - _val_err) < self.min_delta:
                        _patience -= 1
                        if  _epochs > self.min_epoch:
                            if _patience <= 0:
                                print('Early stopping at {}th iteration. Test Acc {}'.format(i, self.test_acc[-1]))
                                break
                    else:
                        _patience = self.patience
                        tmp_best_val_err = _val_err
                    #end if
                #end if


                if i % 10 == 0:
                    print('iter: %d, train error  : %g'%(i, self.train_err[i]))
                    print('iter: %d, test accuracy  : %g'%(i, self.test_acc[i]))

            #end for
            self.early_stop_epoch = _epochs
            self.saver.save(sess, self.save_path)
        #end with

        self.time_taken_one_epoch = (time_to_update/_epochs) * 1000
        return self
    #end def
#end class


def read_data(choice='char', doc_len=MAX_DOCUMENT_LENGTH):
    if choice == 'char': row_num = 1
    else: row_num = 2

    x_train, y_train, x_test, y_test = [], [], [], []
    with open('data/train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[row_num])
            y_train.append(int(row[0]))
        #end for
    #end with

    with open('data/test_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[row_num])
            y_test.append(int(row[0]))
        #end for
    #end with

    x_train, x_val, y_train, y_val = train_test_split(
                                                    x_train,
                                                    y_train,
                                                    test_size=0.2,
                                                    random_state=seed)

    x_train = pd.Series(x_train)
    y_train = pd.Series(y_train)
    x_val = pd.Series(x_val)
    y_val = pd.Series(y_val)
    x_test = pd.Series(x_test)
    y_test = pd.Series(y_test)
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values

    if choice == 'char':
        char_processor = tf.contrib.learn.preprocessing.ByteProcessor(doc_len)
        x_train = np.array(list(char_processor.fit_transform(x_train)))
        x_val = np.array(list(char_processor.transform(x_val)))
        x_test = np.array(list(char_processor.transform(x_test)))
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(doc_len)
        x_train = np.array(list(vocab_processor.fit_transform(x_train)))
        x_val = np.array(list(vocab_processor.transform(x_val)))
        x_test = np.array(list(vocab_processor.transform(x_test)))

        n_words = len(vocab_processor.vocabulary_)
        print('Total words: %d' % n_words)

        return x_train, y_train, x_val, y_val, x_test, y_test, n_words
    #end if
#end def


def arg_cnn_dict(model_save_path, C1_filters=10, C2_filters=10,
            C1_kernal_size=[20, 256], C2_kernal_size=[20, 1],
            S1_window=4, S2_window=4,
            S1_strides=2, S2_strides=2,
            drop_out=False, keep_prob=0.9,
            n_words=None, embedding_size=None, choice='char'):

    C1_dict = dict(filters=C1_filters, kernel_size=C1_kernal_size, padding='VALID')
    C2_dict = dict(filters=C2_filters, kernel_size=C2_kernal_size, padding='VALID')
    S1_dict = dict(window=S1_window, strides=S1_strides, padding='SAME')
    S2_dict = dict(window=S2_window, strides=S2_strides, padding='SAME')
    hidden_layer_dict = dict(C1=C1_dict, C2=C2_dict, S1=S1_dict, S2=S2_dict)

    init_dict = dict(
                save_path=model_save_path,
                input_dim=MAX_DOCUMENT_LENGTH,
                output_dim=MAX_LABEL,
                drop_out=drop_out,
                keep_prob=keep_prob,
                hidden_layer_dict=hidden_layer_dict,
                batch_size=128,
                learning_rate=learning_rate,
                epochs=2000,
                early_stop=True,
                patience=20,
                min_delta = 0.0005,
                n_words=n_words,
                embedding_size=embedding_size,
                choice=choice)
    return init_dict
#end def


def arg_rnn_dict(model_save_path, n_hidden_list, input_dim=MAX_DOCUMENT_LENGTH,
            drop_out=False, keep_prob=0.9,
            n_words=None, embedding_size=None, choice='char', rnn_choice='GRU',
            gradient_clipped=False, patience=20, min_epoch=20):


    init_dict = dict(
                save_path=model_save_path,
                input_dim=input_dim,
                output_dim=MAX_LABEL,
                drop_out=drop_out,
                keep_prob=keep_prob,
                n_hidden_list=n_hidden_list,
                batch_size=128,
                learning_rate=learning_rate,
                epochs=2000,
                early_stop=True,
                patience=patience,
                min_delta=0.0005,
                min_epoch=min_epoch,
                n_words=n_words,
                embedding_size=embedding_size,
                choice=choice,
                rnn_choice=rnn_choice,
                gradient_clipped=gradient_clipped)
    return init_dict
#end def


def plot_figures(df):
    def process_list_col(model_name, col_name):
        l = list(df[df['name'] == model_name][col_name])
        l = l[0][1:-2].split(',')
        return [float(n) for n in l]
    #end def

    # =========================== Q1
    # Plot Train Error
    plt.figure("Q1 Train Error against Epoch")
    plt.title("Q1 Train Error against Epoch")
    train_err = process_list_col('q1_char_cnn', 'train_err')

    plt.plot(range(len(train_err)), train_err)
    plt.xlabel('Epochs')
    plt.ylabel('Train Error')
    plt.grid(b=True)
    #end for
    plt.tight_layout()
    plt.savefig('figures/b/1_train_error_vs_epoch.png')

    # Plot Test Accuracy
    plt.figure("Q1 Test Accuracy against Epoch")
    plt.title("Q1 Test Accuracy against Epoch")
    test_acc = process_list_col('q1_char_cnn', 'test_acc')

    plt.plot(range(len(test_acc)), test_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    #end for
    plt.tight_layout()
    plt.savefig('figures/b/1_test_accuracy_vs_epoch.png')

    # =========================== Q2
    # Plot Train Error
    plt.figure("Q2 Train Error against Epoch")
    plt.title("Q2 Train Error against Epoch")
    train_err = process_list_col('q2_word_cnn', 'train_err')

    plt.plot(range(len(train_err)), train_err)
    plt.xlabel('Epochs')
    plt.ylabel('Train Error')
    plt.grid(b=True)
    #end for
    plt.tight_layout()
    plt.savefig('figures/b/2_train_error_vs_epoch.png')

    # Plot Test Accuracy
    plt.figure("Q2 Test Accuracy against Epoch")
    plt.title("Q2 Test Accuracy against Epoch")
    test_acc = process_list_col('q2_word_cnn', 'test_acc')

    plt.plot(range(len(test_acc)), test_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    #end for
    plt.tight_layout()
    plt.savefig('figures/b/2_test_accuracy_vs_epoch.png')

    # =========================== Q3
    # Plot Train Error
    plt.figure("Q3 Train Error against Epoch")
    plt.title("Q3 Train Error against Epoch")
    train_err = process_list_col('q3_char_rnn', 'train_err')

    plt.plot(range(len(train_err)), train_err)
    plt.xlabel('Epochs')
    plt.ylabel('Train Error')
    plt.grid(b=True)
    #end for
    plt.tight_layout()
    plt.savefig('figures/b/3_train_error_vs_epoch.png')

    # Plot Test Accuracy
    plt.figure("Q3 Test Accuracy against Epoch")
    plt.title("Q3 Test Accuracy against Epoch")
    test_acc = process_list_col('q3_char_rnn', 'test_acc')

    plt.plot(range(len(test_acc)), test_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    #end for
    plt.tight_layout()
    plt.savefig('figures/b/3_test_accuracy_vs_epoch.png')

    # =========================== Q4
    # Plot Train Error
    plt.figure("Q4 Train Error against Epoch")
    plt.title("Q4 Train Error against Epoch")
    train_err = process_list_col('q4_word_rnn', 'train_err')

    plt.plot(range(len(train_err)), train_err)
    plt.xlabel('Epochs')
    plt.ylabel('Train Error')
    plt.grid(b=True)
    #end for
    plt.tight_layout()
    plt.savefig('figures/b/4_train_error_vs_epoch.png')

    # Plot Test Accuracy
    plt.figure("Q4 Test Accuracy against Epoch")
    plt.title("Q4 Test Accuracy against Epoch")
    test_acc = process_list_col('q4_word_rnn', 'test_acc')

    plt.plot(range(len(test_acc)), test_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.grid(b=True)
    #end for
    plt.tight_layout()
    plt.savefig('figures/b/4_test_accuracy_vs_epoch.png')

    # =========================== Q5 part I
    model_list = ['q1_char_cnn', 'q2_word_cnn', 'q3_char_rnn', 'q4_word_rnn']

    # Plot Test Accuracy
    plt.figure("Q5 Test Accuracy against Epoch")
    plt.title("Q5 Test Accuracy against Epoch")
    for val in model_list:
        test_acc = process_list_col(val, 'test_acc')

        plt.plot(range(len(test_acc)), test_acc, label = 'model = {}'.format(val))
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.tight_layout()
    plt.savefig('figures/b/1to5_test_accuracy_vs_epoch.png')

    # Plot Total Training Time
    time_per_epoch = []
    num_epochs = []

    for val in model_list:
        time_per_epoch.append(float(df[df['name'] == val]['time_taken_one_epoch']))
        num_epochs.append(int(df[df['name'] == val]['early_stop_epoch']))

    total_time = [x*y for x,y in zip(time_per_epoch, num_epochs)]
    plt.figure("Q5 Total Time/ms")
    plt.title("Q5 Total Time/ms")
    plt.plot(model_list, total_time)
    # plt.xticks(rotation=90)
    plt.xticks()
    plt.xlabel('Model')
    plt.ylabel('Total Time/ms')
    plt.grid(b=True)
    plt.tight_layout()
    plt.savefig('figures/b/1to5_total_time.png')

    # =========================== Q6a
    char_model_list = ['q6a_char_rnn_BASIC',
                        'q6a_char_rnn_50_BASIC',
                        'q6a_char_rnn_20_BASIC',
                        'q6a_char_rnn_LSTM',
                        'q6a_char_rnn_50_LSTM']

    word_model_list = ['q6a_word_rnn_BASIC',
                        'q6a_word_rnn_50_BASIC',
                        'q6a_word_rnn_20_BASIC',
                        'q6a_word_rnn_LSTM',
                        'q6a_word_rnn_50_LSTM']

    #### plot doc_len = 100
    # Plot Char BASIC Test Accuracy
    plt.figure("Q6a Char BASIC Test Accuracy against Epoch")
    plt.title("Q6a Char BASIC Test Accuracy against Epoch")
    for val in char_model_list[:3]:
        test_acc = process_list_col(val, 'test_acc')

        plt.plot(range(len(test_acc)), test_acc, label = 'model = {}'.format(val))
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.tight_layout()
    plt.savefig('figures/b/6a_char_basic_test_accuracy_vs_epoch.png')

    # Plot Char LSTM Test Accuracy
    plt.figure("Q6a Char LSTM Test Accuracy against Epoch")
    plt.title("Q6a Char LSTM Test Accuracy against Epoch")
    for val in char_model_list[3:]:
        test_acc = process_list_col(val, 'test_acc')

        plt.plot(range(len(test_acc)), test_acc, label = 'model = {}'.format(val))
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.tight_layout()
    plt.savefig('figures/b/6a_char_lstm_test_accuracy_vs_epoch.png')

    #### plot doc_len = 50 and 20
    # Plot Word BASIC Test Accuracy
    plt.figure("Q6a Word BASIC Test Accuracy against Epoch")
    plt.title("Q6a Word BASIC Test Accuracy against Epoch")
    for val in word_model_list[:3]:
        test_acc = process_list_col(val, 'test_acc')

        plt.plot(range(len(test_acc)), test_acc, label = 'model = {}'.format(val))
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.tight_layout()
    plt.savefig('figures/b/6a_word_basic_test_accuracy_vs_epoch.png')

    #### plot doc_len = 50 and 20
    # Plot Word LSTM Test Accuracy
    plt.figure("Q6a Word LSTM Test Accuracy against Epoch")
    plt.title("Q6a Word LSTM Test Accuracy against Epoch")
    for val in word_model_list[3:]:
        test_acc = process_list_col(val, 'test_acc')

        plt.plot(range(len(test_acc)), test_acc, label = 'model = {}'.format(val))
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.tight_layout()
    plt.savefig('figures/b/6a_word_lstm_test_accuracy_vs_epoch.png')

    # =========================== Q6c
    model_list = ['q3_char_rnn', 'q6c_char_rnn',
                    'q4_word_rnn', 'q6c_word_rnn']

    # Plot Training Errors
    plt.figure("Q6c Train Error against Epoch")
    plt.title("Q6c Train Error against Epoch")
    for val in model_list:
        train_err = process_list_col(val, 'train_err')

        plt.plot(range(len(train_err)), train_err, label = 'model = {}'.format(val))
        plt.xlabel('Epochs')
        plt.ylabel('Train Error')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.tight_layout()
    plt.savefig('figures/b/6c_train_error_vs_epoch.png')
#end def


def main():

    x_train, y_train, x_val, y_val, x_test, y_test = read_data(choice='char')
    x_train_word, y_train_word, x_val_word, y_val_word, x_test_word, y_test_word, n_words = read_data(choice='word')

    # x_train = x_train[:100]
    # y_train = y_train[:100]
    # x_test = x_test[:10]
    # y_test = y_test[:10]
    # x_val = x_val[:10]
    # y_val = y_val[:10]

    # x_train_word = x_train_word[:100]
    # y_train_word = y_train_word[:100]
    # x_test_word = x_test_word[:10]
    # y_test_word = y_test_word[:10]
    # x_val_word = x_val_word[:10]
    # y_val_word = y_val_word[:10]

    result_dict_list = list()

    # =========================== Q1
    print('='*100)
    print('Q1 Char CNN')
    print('='*100)

    tf.reset_default_graph()
    init_dict = arg_cnn_dict(
                    model_save_path='models/b/1_char_cnn',
                    C1_filters=10, C2_filters=10,
                    C1_kernal_size=[20, 256], C2_kernal_size=[20, 1],
                    S1_window=4, S2_window=4,
                    S1_strides=2, S2_strides=2,
                    drop_out=False, keep_prob=1,
                    choice='char')

    char_cnn = CNNClassifer(**init_dict).train(
                                X_train=x_train,
                                Y_train=y_train,
                                X_test=x_test,
                                Y_test=y_test,
                                X_val=x_val,
                                Y_val=y_val)

    train_err, test_acc, time_taken_one_epoch, early_stop_epoch = char_cnn.train_err, char_cnn.test_acc, char_cnn.time_taken_one_epoch, char_cnn.early_stop_epoch

    _result_dict = dict(name='q1_char_cnn', train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
    result_dict_list.append(_result_dict)

    # =========================== Q2
    print()
    print()
    print('='*100)
    print('Q2 Word CNN')
    print('='*100)

    tf.reset_default_graph()
    init_dict = arg_cnn_dict(
                    model_save_path='models/b/2_word_cnn',
                    C1_filters=10, C2_filters=10,
                    C1_kernal_size=[20, 20], C2_kernal_size=[20, 1],
                    S1_window=4, S2_window=4,
                    S1_strides=2, S2_strides=2,
                    drop_out=False, keep_prob=1,
                    n_words=n_words, embedding_size=20,
                    choice='word')

    word_cnn = CNNClassifer(**init_dict).train(
                                X_train=x_train_word,
                                Y_train=y_train_word,
                                X_test=x_test_word,
                                Y_test=y_test_word,
                                X_val=x_val_word,
                                Y_val=y_val_word)

    train_err, test_acc, time_taken_one_epoch, early_stop_epoch = word_cnn.train_err, word_cnn.test_acc, word_cnn.time_taken_one_epoch, word_cnn.early_stop_epoch

    _result_dict = dict(name='q2_word_cnn', train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
    result_dict_list.append(_result_dict)

    # =========================== Q3
    print()
    print()
    print('='*100)
    print('Q3 Char RNN')
    print('='*100)

    tf.reset_default_graph()
    init_dict = arg_rnn_dict(
                    model_save_path='models/b/3_char_rnn',
                    n_hidden_list=[20],
                    drop_out=False, keep_prob=1,
                    choice='char',
                    rnn_choice='GRU')

    char_rnn = RNNClassifer(**init_dict).train(
                                X_train=x_train,
                                Y_train=y_train,
                                X_test=x_test,
                                Y_test=y_test,
                                X_val=x_val,
                                Y_val=y_val)
    train_err, test_acc, time_taken_one_epoch, early_stop_epoch = char_rnn.train_err, char_rnn.test_acc, char_rnn.time_taken_one_epoch, char_rnn.early_stop_epoch

    _result_dict = dict(name='q3_char_rnn', train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
    result_dict_list.append(_result_dict)

    # =========================== Q4
    print()
    print()
    print('='*100)
    print('Q4 Word RNN')
    print('='*100)

    tf.reset_default_graph()
    init_dict = arg_rnn_dict(
                    model_save_path='models/b/4_word_rnn',
                    n_hidden_list=[20],
                    drop_out=False, keep_prob=1,
                    choice='word', n_words=n_words, embedding_size=20,
                    rnn_choice='GRU')

    word_rnn = RNNClassifer(**init_dict).train(
                                X_train=x_train_word,
                                Y_train=y_train_word,
                                X_test=x_test_word,
                                Y_test=y_test_word,
                                X_val=x_val_word,
                                Y_val=y_val_word)
    train_err, test_acc, time_taken_one_epoch, early_stop_epoch = word_rnn.train_err, word_rnn.test_acc, word_rnn.time_taken_one_epoch, word_rnn.early_stop_epoch

    _result_dict = dict(name='q4_word_rnn', train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
    result_dict_list.append(_result_dict)

    # =========================== Q5
    print()
    print()
    print('='*100)
    print('Q5 Char/Word CNN/RNN on Keep Prob - 0.1, 0.3, 0.5, 0.7, 0.9')
    print('='*100)

    keep_probs = [0.1, 0.3, 0.5, 0.7, 0.9]

    for prob in keep_probs:
        print('-'*40)
        print(prob)
        print('-'*40)

        # char CNN with dropout
        print('Char CNN')
        tf.reset_default_graph()
        init_dict = arg_cnn_dict(
                        model_save_path='models/b/5_char_cnn_' + str(prob),
                        C1_filters=10, C2_filters=10,
                        C1_kernal_size=[20, 256], C2_kernal_size=[20, 1],
                        S1_window=4, S2_window=4,
                        S1_strides=2, S2_strides=2,
                        drop_out=True, keep_prob=prob,
                        choice='char')

        char_cnn = CNNClassifer(**init_dict).train(
                                    X_train=x_train,
                                    Y_train=y_train,
                                    X_test=x_test,
                                    Y_test=y_test,
                                    X_val=x_val,
                                    Y_val=y_val)
        train_err, test_acc, time_taken_one_epoch, early_stop_epoch = char_cnn.train_err, char_cnn.test_acc, char_cnn.time_taken_one_epoch, char_cnn.early_stop_epoch

        _result_dict = dict(name='q5_char_cnn_'+str(prob), train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
        result_dict_list.append(_result_dict)

        print()

        # word CNN with dropout
        tf.reset_default_graph()
        print('Word CNN')
        init_dict = arg_cnn_dict(
                        model_save_path='models/b/5_word_cnn_' + str(prob),
                        C1_filters=10, C2_filters=10,
                        C1_kernal_size=[20, 20], C2_kernal_size=[20, 1],
                        S1_window=4, S2_window=4,
                        S1_strides=2, S2_strides=2,
                        drop_out=True, keep_prob=prob,
                        n_words=n_words, embedding_size=20,
                        choice='word')

        word_cnn = CNNClassifer(**init_dict).train(
                                    X_train=x_train_word,
                                    Y_train=y_train_word,
                                    X_test=x_test_word,
                                    Y_test=y_test_word,
                                    X_val=x_val_word,
                                    Y_val=y_val_word)
        train_err, test_acc, time_taken_one_epoch, early_stop_epoch = word_cnn.train_err, word_cnn.test_acc, word_cnn.time_taken_one_epoch, word_cnn.early_stop_epoch

        _result_dict = dict(name='q5_word_cnn_'+str(prob), train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
        result_dict_list.append(_result_dict)

        print()

        # char RNN with dropout
        tf.reset_default_graph()
        print('Char RNN')
        init_dict = arg_rnn_dict(
                        model_save_path='models/b/5_char_rnn_' + str(prob),
                        n_hidden_list=[20],
                        drop_out=True, keep_prob=prob,
                        choice='char',
                        rnn_choice='GRU')

        char_rnn = RNNClassifer(**init_dict).train(
                                    X_train=x_train,
                                    Y_train=y_train,
                                    X_test=x_test,
                                    Y_test=y_test,
                                    X_val=x_val,
                                    Y_val=y_val)
        train_err, test_acc, time_taken_one_epoch, early_stop_epoch = char_rnn.train_err, char_rnn.test_acc, char_rnn.time_taken_one_epoch, char_rnn.early_stop_epoch

        _result_dict = dict(name='q5_char_rnn_'+str(prob), train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
        result_dict_list.append(_result_dict)

        print()

        # word RNN with dropout
        tf.reset_default_graph()
        print('Word RNN')
        init_dict = arg_rnn_dict(
                        model_save_path='models/b/5_word_rnn_' + str(prob),
                        n_hidden_list=[20],
                        drop_out=True, keep_prob=prob,
                        choice='word', n_words=n_words, embedding_size=20,
                        rnn_choice='GRU')

        word_rnn = RNNClassifer(**init_dict).train(
                                    X_train=x_train_word,
                                    Y_train=y_train_word,
                                    X_test=x_test_word,
                                    Y_test=y_test_word,
                                    X_val=x_val_word,
                                    Y_val=y_val_word)
        train_err, test_acc, time_taken_one_epoch, early_stop_epoch = word_rnn.train_err, word_rnn.test_acc, word_rnn.time_taken_one_epoch, word_rnn.early_stop_epoch
        _result_dict = dict(name='q5_word_rnn_'+str(prob), train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
        result_dict_list.append(_result_dict)
    #end for

    # =========================== Q6 a
    print()
    print()
    print('='*100)
    print('Q6a Char/Word RNN w/ LSTM/BASIC RNN')
    print('='*100)

    rnn_choices = ['BASIC', 'LSTM']
    for rnn_choice in rnn_choices:
        print('-'*40)
        print(rnn_choice)
        print('-'*40)

        tf.reset_default_graph()
        print('Char RNN')
        init_dict = arg_rnn_dict(
                        model_save_path='models/b/6_char_rnn_' + rnn_choice,
                        input_dim=20,
                        n_hidden_list=[20],
                        drop_out=False, keep_prob=1,
                        choice='char',
                        rnn_choice=rnn_choice)

        char_rnn = RNNClassifer(**init_dict).train(
                                    X_train=x_train,
                                    Y_train=y_train,
                                    X_test=x_test,
                                    Y_test=y_test,
                                    X_val=x_val,
                                    Y_val=y_val)
        train_err, test_acc, time_taken_one_epoch, early_stop_epoch = char_rnn.train_err, char_rnn.test_acc, char_rnn.time_taken_one_epoch, char_rnn.early_stop_epoch
        _result_dict = dict(name='q6a_char_rnn_'+rnn_choice, train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
        result_dict_list.append(_result_dict)

        print()

        tf.reset_default_graph()
        print('Word RNN')
        init_dict = arg_rnn_dict(
                        model_save_path='models/b/6_word_rnn_' + rnn_choice,
                        n_hidden_list=[20],
                        drop_out=False, keep_prob=1,
                        choice='word', n_words=n_words, embedding_size=20,
                        rnn_choice=rnn_choice)

        word_rnn = RNNClassifer(**init_dict).train(
                                    X_train=x_train_word,
                                    Y_train=y_train_word,
                                    X_test=x_test_word,
                                    Y_test=y_test_word,
                                    X_val=x_val_word,
                                    Y_val=y_val_word)
        train_err, test_acc, time_taken_one_epoch, early_stop_epoch = word_rnn.train_err, word_rnn.test_acc, word_rnn.time_taken_one_epoch, word_rnn.early_stop_epoch
        _result_dict = dict(name='q6a_word_rnn_'+rnn_choice, train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
        result_dict_list.append(_result_dict)
    #end for

    # Suspect underfitting
    # Try document length of 50 instead of 100
    x_train_50, y_train_50, x_val_50, y_val_50, x_test_50, y_test_50 = read_data(choice='char', doc_len=50)
    x_train_word_50, y_train_word_50, x_val_word_50, y_val_word_50, x_test_word_50, y_test_word_50, n_words = read_data(choice='word', doc_len=50)
    for rnn_choice in rnn_choices:
        print('-'*40)
        print('Doc Len 50, ' + rnn_choice)
        print('-'*40)

        tf.reset_default_graph()
        print('Doc Len 50 Char RNN')
        init_dict = arg_rnn_dict(
                        model_save_path='models/b/6_char_rnn_50_' + rnn_choice,
                        input_dim=50,
                        n_hidden_list=[20],
                        drop_out=False, keep_prob=1,
                        choice='char',
                        rnn_choice=rnn_choice)

        char_rnn = RNNClassifer(**init_dict).train(
                                    X_train=x_train_50,
                                    Y_train=y_train_50,
                                    X_test=x_test_50,
                                    Y_test=y_test_50,
                                    X_val=x_val_50,
                                    Y_val=y_val_50)
        train_err, test_acc, time_taken_one_epoch, early_stop_epoch = char_rnn.train_err, char_rnn.test_acc, char_rnn.time_taken_one_epoch, char_rnn.early_stop_epoch
        _result_dict = dict(name='q6a_char_rnn_50_'+rnn_choice, train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
        result_dict_list.append(_result_dict)

        print()

        tf.reset_default_graph()
        print('Doc Len 50 Word RNN')
        init_dict = arg_rnn_dict(
                        model_save_path='models/b/6_word_rnn_50_' + rnn_choice,
                        input_dim=50,
                        n_hidden_list=[20],
                        drop_out=False, keep_prob=1,
                        choice='word', n_words=n_words, embedding_size=20,
                        rnn_choice=rnn_choice)

        word_rnn = RNNClassifer(**init_dict).train(
                                    X_train=x_train_word_50,
                                    Y_train=y_train_word_50,
                                    X_test=x_test_word_50,
                                    Y_test=y_test_word_50,
                                    X_val=x_val_word_50,
                                    Y_val=y_val_word_50)
        train_err, test_acc, time_taken_one_epoch, early_stop_epoch = word_rnn.train_err, word_rnn.test_acc, word_rnn.time_taken_one_epoch, word_rnn.early_stop_epoch
        _result_dict = dict(name='q6a_word_rnn_50_'+rnn_choice, train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
        result_dict_list.append(_result_dict)
    #end for

    # Suspect underfitting
    # Try document length of 20 for Basic RNN instead of 100
    rnn_choices = ['BASIC']
    x_train_20, y_train_20, x_val_20, y_val_20, x_test_20, y_test_20 = read_data(choice='char', doc_len=20)
    x_train_word_20, y_train_word_20, x_val_word_20, y_val_word_20, x_test_word_20, y_test_word_20, n_words = read_data(choice='word', doc_len=20)
    for rnn_choice in rnn_choices:
        print('-'*40)
        print('Doc Len 20, ' + rnn_choice)
        print('-'*40)

        tf.reset_default_graph()
        print('Doc Len 20 Char RNN')
        init_dict = arg_rnn_dict(
                        model_save_path='models/b/6_char_rnn_20_' + rnn_choice,
                        input_dim=20,
                        n_hidden_list=[20],
                        drop_out=False, keep_prob=1,
                        choice='char',
                        rnn_choice=rnn_choice)

        char_rnn = RNNClassifer(**init_dict).train(
                                    X_train=x_train_20,
                                    Y_train=y_train_20,
                                    X_test=x_test_20,
                                    Y_test=y_test_20,
                                    X_val=x_val_20,
                                    Y_val=y_val_20)
        train_err, test_acc, time_taken_one_epoch, early_stop_epoch = char_rnn.train_err, char_rnn.test_acc, char_rnn.time_taken_one_epoch, char_rnn.early_stop_epoch
        _result_dict = dict(name='q6a_char_rnn_20_'+rnn_choice, train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
        result_dict_list.append(_result_dict)

        print()

        tf.reset_default_graph()
        print('Doc Len 20 Word RNN')
        init_dict = arg_rnn_dict(
                        model_save_path='models/b/6_word_rnn_20_' + rnn_choice,
                        input_dim=20,
                        n_hidden_list=[20],
                        drop_out=False, keep_prob=1,
                        choice='word', n_words=n_words, embedding_size=20,
                        rnn_choice=rnn_choice)

        word_rnn = RNNClassifer(**init_dict).train(
                                    X_train=x_train_word_20,
                                    Y_train=y_train_word_20,
                                    X_test=x_test_word_20,
                                    Y_test=y_test_word_20,
                                    X_val=x_val_word_20,
                                    Y_val=y_val_word_20)
        train_err, test_acc, time_taken_one_epoch, early_stop_epoch = word_rnn.train_err, word_rnn.test_acc, word_rnn.time_taken_one_epoch, word_rnn.early_stop_epoch
        _result_dict = dict(name='q6a_word_rnn_20_'+rnn_choice, train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
        result_dict_list.append(_result_dict)
    #end for

    # =========================== Q6 b
    print()
    print()
    print('='*100)
    print('Q6b Char/Word RNN w/ 2-layer')
    print('='*100)

    tf.reset_default_graph()
    print('Char RNN')
    init_dict = arg_rnn_dict(
                    model_save_path='models/b/6_char_rnn_2_layer',
                    n_hidden_list=[20, 20],
                    drop_out=False, keep_prob=1,
                    choice='char',
                    rnn_choice='GRU')

    char_rnn = RNNClassifer(**init_dict).train(
                                X_train=x_train,
                                Y_train=y_train,
                                X_test=x_test,
                                Y_test=y_test,
                                X_val=x_val,
                                Y_val=y_val)
    train_err, test_acc, time_taken_one_epoch, early_stop_epoch = char_rnn.train_err, char_rnn.test_acc, char_rnn.time_taken_one_epoch, char_rnn.early_stop_epoch
    _result_dict = dict(name='q6b_char_rnn', train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
    result_dict_list.append(_result_dict)

    print()

    tf.reset_default_graph()
    print('Word RNN')
    init_dict = arg_rnn_dict(
                    model_save_path='models/b/6_word_rnn_2_layer',
                    n_hidden_list=[20, 20],
                    drop_out=False, keep_prob=1,
                    choice='word', n_words=n_words, embedding_size=20,
                    rnn_choice='GRU')

    word_rnn = RNNClassifer(**init_dict).train(
                                X_train=x_train_word,
                                Y_train=y_train_word,
                                X_test=x_test_word,
                                Y_test=y_test_word,
                                X_val=x_val_word,
                                Y_val=y_val_word)
    train_err, test_acc, time_taken_one_epoch, early_stop_epoch = word_rnn.train_err, word_rnn.test_acc, word_rnn.time_taken_one_epoch, word_rnn.early_stop_epoch

    _result_dict = dict(name='q6b_word_rnn', train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
    result_dict_list.append(_result_dict)

    # =========================== Q6 c
    print()
    print()
    print('='*100)
    print('Q6c. Char/Word RNN w/ Gradient Clipped at Threshold 2.0')
    print('='*100)

    tf.reset_default_graph()
    print('Char RNN')
    init_dict = arg_rnn_dict(
                    model_save_path='models/b/6_char_rnn_clipped',
                    n_hidden_list=[20, 20],
                    drop_out=False, keep_prob=1,
                    choice='char',
                    rnn_choice='GRU',
                    gradient_clipped=True)

    char_rnn = RNNClassifer(**init_dict).train(
                                X_train=x_train,
                                Y_train=y_train,
                                X_test=x_test,
                                Y_test=y_test,
                                X_val=x_val,
                                Y_val=y_val)
    train_err, test_acc, time_taken_one_epoch, early_stop_epoch = char_rnn.train_err, char_rnn.test_acc, char_rnn.time_taken_one_epoch, char_rnn.early_stop_epoch

    _result_dict = dict(name='q6c_char_rnn', train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
    result_dict_list.append(_result_dict)

    print()

    tf.reset_default_graph()
    print('Word RNN')
    init_dict = arg_rnn_dict(
                    model_save_path='models/b/6_word_rnn_clipped',
                    n_hidden_list=[20, 20],
                    drop_out=False, keep_prob=1,
                    choice='word', n_words=n_words, embedding_size=20,
                    rnn_choice='GRU',
                    gradient_clipped=True)

    word_rnn = RNNClassifer(**init_dict).train(
                                X_train=x_train_word,
                                Y_train=y_train_word,
                                X_test=x_test_word,
                                Y_test=y_test_word,
                                X_val=x_val_word,
                                Y_val=y_val_word)
    train_err, test_acc, time_taken_one_epoch, early_stop_epoch = word_rnn.train_err, word_rnn.test_acc, word_rnn.time_taken_one_epoch, word_rnn.early_stop_epoch
    _result_dict = dict(name='q6c_word_rnn', train_err=train_err, test_acc=test_acc, time_taken_one_epoch=time_taken_one_epoch, early_stop_epoch=early_stop_epoch)
    result_dict_list.append(_result_dict)

    df = pd.DataFrame.from_dict(result_dict_list)
    df.to_csv('./csv_results/q2.csv')

    _df = pd.read_csv('./csv_results/q2.csv')
    plot_figures(_df)
#end def


if __name__ == '__main__':
  main()