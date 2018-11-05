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
FILTER_SHAPE1_CHAR = [20, 256]
FILTER_SHAPE2 = [20, 1]
FILTER_SHAPE1_WORD = [20,20]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
BATCH_SIZE = 128
HIDDEN_SIZE = 20
CHAR_SIZE = 256
EMBEDDING_SIZE = 20

no_epochs = 1000
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def cnn_model(x_, model_decision = 'char', drop_out=False, _keep_prob=None):

    with tf.variable_scope('CNN_Layer1'):
        if model_decision == 'char':
            conv1 = tf.layers.conv2d(
                x_,
                filters=N_FILTERS,
                kernel_size=FILTER_SHAPE1_CHAR,
                padding='VALID',
                activation=tf.nn.relu)
        else:
            conv1 = tf.layers.conv2d(
                x_,
                filters=N_FILTERS,
                kernel_size=FILTER_SHAPE1_WORD,
                padding='VALID',
                activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')

        if drop_out:
            pool1 = tf.nn.dropout(pool1, _keep_prob)

    #Pool TO ASK
    # pool1 = tf.squeeze(tf.reduce_max(pool1, 1), squeeze_dims=[1])

    with tf.variable_scope('CNN_Layer2'):
        conv2 = tf.layers.conv2d(
            pool1,
            filters=N_FILTERS,
            kernel_size=FILTER_SHAPE2,
            padding='VALID',
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')
        if drop_out:
            pool2 = tf.nn.dropout(pool2, _keep_prob)

    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

    logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

    return logits

def run_cnn_model(x_train, y_train, x_val, y_val, x_test, y_test,
                    model_decision='char',
                    drop_out = False, keep_prob = None,
                    min_delta = 0.0005, patience = 20):
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    if drop_out:
        _keep_prob = tf.placeholder(tf.float32)
    else:
        _keep_prob = None

    if model_decision == 'char':
        x_ = tf.reshape(tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

    elif model_decision == 'word':

        x_ = tf.contrib.layers.embed_sequence(
                                                x,
                                                vocab_size= n_words,
                                                embed_dim= EMBEDDING_SIZE)
        x_ = tf.reshape(x_, [-1, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE, 1])


    logits = cnn_model(x_,model_decision,drop_out,_keep_prob)

    #entropy
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))

    #accuracy
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y_, MAX_LABEL), 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # Optimizer
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)
    np.random.seed(10)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # training
        train_err = []
        test_acc = []
        time_to_update = 0
        t = time.time()
        N = len(x_train)
        idx = np.arange(N)
        tmp_best_val_err = 1000
        _epoch = 0
        _patience = patience
        for i in range(no_epochs):
            _epoch += 1
            np.random.shuffle(idx)
            x_train = x_train[idx]
            y_train = y_train[idx]

            t = time.time()
            for _start, _end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
                if drop_out:
                    train_op.run(feed_dict={x: x_train[_start:_end], y_: y_train[_start:_end], _keep_prob:keep_prob})
                else:
                    train_op.run(feed_dict={x: x_train[_start:_end], y_: y_train[_start:_end]})
            time_to_update += (time.time() - t)

            if drop_out:
                train_err.append(entropy.eval(feed_dict={x: x_train, y_: y_train, _keep_prob:keep_prob}))
                _val_err = entropy.eval(feed_dict={x: x_val, y_: y_val, _keep_prob:keep_prob})
                test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test, _keep_prob:1.0}))

            else:
                train_err.append(entropy.eval(feed_dict={x: x_train, y_: y_train}))
                _val_err = entropy.eval(feed_dict={x: x_val, y_: y_val})
                test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

            if (tmp_best_val_err - _val_err) < min_delta:
                _patience -= 1

                if _patience <= 0:
                    print('Early stopping at {}th iteration. Test Acc {}'.format(i, test_acc[-1]))
                    break
            else:
                _patience = patience
                tmp_best_val_err = _val_err

    sess.close()
    return train_err,test_acc,time_to_update, _epoch

def rnn_model(x,_keep_prob =None,drop_out = False,model_decision='char',rnn_decision='GRU',rnn_layer = 1):

    if model_decision == 'char':
        byte_vectors = tf.one_hot(x, CHAR_SIZE)

        if rnn_layer == 1:
            if rnn_decision == 'GRU':
                cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
                outputs, states = tf.nn.dynamic_rnn(cell, byte_vectors, dtype=tf.float32)
            elif rnn_decision == 'BASIC':
                cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
                outputs, states = tf.nn.dynamic_rnn(cell, byte_vectors, dtype=tf.float32)
            elif rnn_decision == 'LSTM':
                cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
                outputs, states = tf.nn.dynamic_rnn(cell, byte_vectors, dtype=tf.float32)
                states = states.h

        elif rnn_layer == 2:
            if rnn_decision == 'GRU':
                # cell1 = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, reuse=tf.get_variable_scope().reuse)
                cell1 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
                cell2 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
                cells = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
                outputs, states = tf.nn.dynamic_rnn(cells, byte_vectors, dtype=tf.float32)

        if isinstance(states, tuple):
            states = states[-1]

        if drop_out:
            states = tf.nn.dropout(states, _keep_prob)

        logits = tf.layers.dense(states, MAX_LABEL, activation=None)

        return logits

    else:
        word_vectors = tf.contrib.layers.embed_sequence(
            x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
        word_vectors = tf.reshape(word_vectors, [-1, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE])


        if rnn_decision == 'GRU':
            cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
            outputs, states = tf.nn.dynamic_rnn(cell, word_vectors , dtype=tf.float32)
        elif rnn_decision == 'BASIC':
            cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
            outputs, states = tf.nn.dynamic_rnn(cell, word_vectors, dtype=tf.float32)
        elif rnn_decision == 'LSTM':
            cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
            outputs, states = tf.nn.dynamic_rnn(cell, word_vectors, dtype=tf.float32)
            states = states.h

        elif rnn_layer == 2:
            if rnn_decision == 'GRU':
                # cell1 = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, reuse=tf.get_variable_scope().reuse)
                cell1 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
                cell2 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
                cells = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
                outputs, states = tf.nn.dynamic_rnn(cells, word_vectors, dtype=tf.float32)

        if isinstance(states, tuple):
            states = states[-1]

        if drop_out:
            states = tf.nn.dropout(states, _keep_prob)

        logits = tf.layers.dense(states, MAX_LABEL, activation=None)

        return logits

def run_rnn_model(x_train, y_train, x_val, y_val, x_test, y_test,
                    model_decision='char',rnn_decision='GRU',
                    drop_out = False, keep_prob = None,
                    gradient_clipped = False,
                    rnn_layer = 1,
                    min_delta = 0.0005, patience = 20):

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    if drop_out:
        _keep_prob = tf.placeholder(tf.float32)
    else:
        _keep_prob = None

    logits = rnn_model(x,keep_prob,drop_out,model_decision,rnn_decision,rnn_layer)

    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))

    #accuracy
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y_, MAX_LABEL), 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    if gradient_clipped:
        # Minimizer
        minimizer = tf.train.AdamOptimizer(lr)
        grads_and_vars = minimizer.compute_gradients(entropy)

        # Gradient clipping
        grad_clipping = tf.constant(2.0, name="grad_clipping")
        clipped_grads_and_vars = []
        for grad, var in grads_and_vars:
            clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
            clipped_grads_and_vars.append((clipped_grad, var))
            # Gradient updates
        train_op = minimizer.apply_gradients(clipped_grads_and_vars)

    else:
        train_op = tf.train.AdamOptimizer(lr).minimize(entropy)
	
    np.random.seed(10)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # training
        train_err = []
        test_acc = []
        time_to_update = 0
        t = time.time()
        N = len(x_train)
        idx = np.arange(N)
        tmp_best_val_err = 1000
        _epoch = 0
        _patience = patience
        for i in range(no_epochs):
            _epoch += 1
            np.random.shuffle(idx)
            x_train = x_train[idx]
            y_train = y_train[idx]

            t = time.time()
            # for e in range(no_epochs):
            #     word_list_, _, loss_  = sess.run([word_list, train_op, entropy], {x: x_train, y_: y_train})
            #     loss.append(loss_)

            for _start, _end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
                if drop_out:
                    train_op.run(feed_dict={x: x_train[_start:_end], y_: y_train[_start:_end], _keep_prob:keep_prob})
                else:
                    train_op.run(feed_dict={x: x_train[_start:_end], y_: y_train[_start:_end]})

            time_to_update += (time.time() - t)

            if drop_out:
                train_err.append(entropy.eval(feed_dict={x: x_train, y_: y_train, _keep_prob:keep_prob}))
                _val_err = entropy.eval(feed_dict={x: x_val, y_: y_val, _keep_prob:keep_prob})
                test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test, _keep_prob:1.0}))

            else:
                train_err.append(entropy.eval(feed_dict={x: x_train, y_: y_train}))
                _val_err = entropy.eval(feed_dict={x: x_val, y_: y_val})
                test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

            if (tmp_best_val_err - _val_err) < min_delta:
                _patience -= 1

                if _patience <= 0:
                    print('Early stopping at {}th iteration. Test Acc {}'.format(i, test_acc[-1]))
                    break
            else:
                _patience = patience
                tmp_best_val_err = _val_err

    sess.close()
    return train_err,test_acc,time_to_update, _epoch

def read_data(choice='char'):

    if choice == 'char': row_num = 1
    else: row_num = 2
    x_train, y_train, x_test, y_test = [], [], [], []
    with open('Assignment2/data/train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[row_num])
            y_train.append(int(row[0]))
        #end for
    #end with

    with open('Assignment2/data/test_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[row_num])
            y_test.append(int(row[0]))

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
        processor =  tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
    else:
        processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)

    x_train = np.array(list(processor.fit_transform(x_train)))
    x_val = np.array(list(processor.transform(x_val)))
    x_test = np.array(list(processor.transform(x_test)))

    if choice == 'char':
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        n_words = len(processor.vocabulary_)
        return x_train, y_train, x_val, y_val, x_test, y_test, n_words

def main():
    global n_words
    x_train, y_train, x_val, y_val, x_test, y_test = read_data('char')
    x_train_word, y_train_word, x_val_word, y_val_word, x_test_word, y_test_word, n_words = read_data('word')
    train_err_dict = dict()
    test_acc_dict = dict()
    time_to_update_dict = dict()
    epochs_dict = dict()

    init_dict_char_cnn = dict(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test,
                            model_decision='char',
                            drop_out = False, keep_prob = None,
                            min_delta = 0.0005, patience = 20)
							
    init_dict_word_cnn = dict(x_train=x_train_word, y_train=y_train_word, x_val=x_val_word, y_val=y_val_word, x_test=x_test_word, y_test=y_test_word,
                            model_decision='word',
                            drop_out = False, keep_prob = None,
                            min_delta = 0.0005, patience = 20)


    init_dict_char_rnn = dict(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test,
                            model_decision='char',rnn_decision='GRU',
                            drop_out = False, keep_prob = None,
                            gradient_clipped = False, rnn_layer = 1,
                            min_delta = 0.0005, patience = 20)

    init_dict_word_rnn = dict(x_train=x_train_word, y_train=y_train_word, x_val=x_val_word, y_val=y_val_word, x_test=x_test_word, y_test=y_test_word,
                            model_decision='word',rnn_decision='GRU',
                            drop_out = False, keep_prob = None,
                            gradient_clipped = False, rnn_layer = 1,
                            min_delta = 0.0005, patience = 20)

    init_dict_char_cnn_backup = init_dict_char_cnn.copy()
    init_dict_word_cnn_backup = init_dict_word_cnn.copy()
    init_dict_char_rnn_backup = init_dict_char_rnn.copy()
    init_dict_word_rnn_backup = init_dict_word_rnn.copy()

    #====Q1====

    tf.reset_default_graph()
    print('training char_cnn_no_dropout:')
    train_err,test_acc,time_to_update,epochs = run_cnn_model(**init_dict_char_cnn)
    train_err_dict['char_cnn_no_dropout'] = train_err
    test_acc_dict['char_cnn_no_dropout'] = test_acc
    time_to_update_dict['char_cnn_no_dropout'] = time_to_update
    epochs_dict['char_cnn_no_dropout'] = epochs
    print("="*50)

    #====Q2====
    tf.reset_default_graph()
    print('training word_cnn_no_dropout:')
    train_err,test_acc,time_to_update,epochs = run_cnn_model(**init_dict_word_cnn)
    train_err_dict['word_cnn_no_dropout'] = train_err
    test_acc_dict['word_cnn_no_dropout'] = test_acc
    time_to_update_dict['word_cnn_no_dropout'] = time_to_update
    epochs_dict['word_cnn_no_dropout'] = epochs
    print("="*50)


    #====Q3====
    tf.reset_default_graph()
    print('training char_rnn_no_dropout:')
    train_err,test_acc,time_to_update,epochs = run_rnn_model(**init_dict_char_rnn)
    train_err_dict['char_rnn_no_dropout'] = train_err
    test_acc_dict['char_rnn_no_dropout'] = test_acc
    time_to_update_dict['char_rnn_no_dropout'] = time_to_update
    epochs_dict['char_rnn_no_dropout'] = epochs
    print("="*50)

    #====Q4====
    tf.reset_default_graph()
    print('training word_rnn_no_dropout:')
    train_err,test_acc,time_to_update,epochs = run_rnn_model(**init_dict_word_rnn)
    train_err_dict['word_rnn_no_dropout'] = train_err
    test_acc_dict['word_rnn_no_dropout'] = test_acc
    time_to_update_dict['word_rnn_no_dropout'] = time_to_update
    epochs_dict['word_rnn_no_dropout'] = epochs
    print("="*50)


    #====Q5====

    init_dict_char_cnn['drop_out'] = True
    init_dict_word_cnn['drop_out'] = True
    init_dict_char_rnn['drop_out'] = True
    init_dict_word_rnn['drop_out'] = True
    for keep_prob in [0.1,0.3,0.5,0.7,0.9]:
        init_dict_char_cnn['keep_prob'] = keep_prob
        init_dict_word_cnn['keep_prob'] = keep_prob
        init_dict_char_rnn['keep_prob'] = keep_prob
        init_dict_word_rnn['keep_prob'] = keep_prob

        #char_cnn
        tf.reset_default_graph()
        print('training char_cnn_with_dropout_{}:'.format(keep_prob))
        train_err,test_acc,time_to_update,epochs = run_rnn_model(**init_dict_char_cnn)
        dict_key = 'char_cnn_' + 'dropout_' + str(keep_prob)
        train_err_dict[dict_key] = train_err
        test_acc_dict[dict_key] = test_acc
        time_to_update_dict[dict_key] = time_to_update
        epochs_dict[dict_key] = epochs
        print("="*50)

        #word_cnn
        tf.reset_default_graph()
        print('training word_cnn_with_dropout_{}:'.format(keep_prob))
        train_err,test_acc,time_to_update,epochs = run_rnn_model(**init_dict_word_cnn)
        dict_key = 'word_cnn_' + 'dropout_' + str(keep_prob)
        train_err_dict[dict_key] = train_err
        test_acc_dict[dict_key] = test_acc
        time_to_update_dict[dict_key] = time_to_update
        epochs_dict[dict_key] = epochs
        print("="*50)

        #char_rnn
        tf.reset_default_graph()
        print('training char_rnn_with_dropout_{}:'.format(keep_prob))
        train_err,test_acc,time_to_update,epochs = run_rnn_model(**init_dict_char_rnn)
        dict_key = 'char_rnn_' + 'dropout_' + str(keep_prob)
        train_err_dict[dict_key] = train_err
        test_acc_dict[dict_key] = test_acc
        time_to_update_dict[dict_key] = time_to_update
        epochs_dict[dict_key] = epochs
        print("="*50)

        #word_rnn
        tf.reset_default_graph()
        print('training word_rnn_with_dropout_{}:'.format(keep_prob))
        train_err,test_acc,time_to_update,epochs = run_rnn_model(**init_dict_word_rnn)
        dict_key = 'word_rnn_' + 'dropout_' + str(keep_prob)
        train_err_dict[dict_key] = train_err
        test_acc_dict[dict_key] = test_acc
        time_to_update_dict[dict_key] = time_to_update
        epochs_dict[dict_key] = epochs
        print("="*50)

    #====Q6a====
    init_dict_char_rnn = init_dict_char_rnn_backup.copy()
    init_dict_word_rnn = init_dict_word_rnn_backup.copy()

    for rnn_decision in ['BASIC','LSTM']:
        init_dict_char_rnn['rnn_decision'] = rnn_decision
        #char_rnn
        tf.reset_default_graph()
        print('training char_rnn_with_{}:'.format(rnn_decision))
        train_err,test_acc,time_to_update,epochs = run_rnn_model(**init_dict_char_rnn)
        dict_key = 'char_rnn_' + 'rnn_decision' + str(rnn_decision)
        train_err_dict[dict_key] = train_err
        test_acc_dict[dict_key] = test_acc
        time_to_update_dict[dict_key] = time_to_update
        epochs_dict[dict_key] = epochs
        print("="*50)

        init_dict_word_rnn['rnn_decision'] = rnn_decision
        #word_rnn
        tf.reset_default_graph()
        print('training word_rnn_with_{}:'.format(rnn_decision))
        train_err,test_acc,time_to_update,epochs = run_rnn_model(**init_dict_word_rnn)
        dict_key = 'word_rnn_' + 'rnn_decision' + str(rnn_decision)
        train_err_dict[dict_key] = train_err
        test_acc_dict[dict_key] = test_acc
        time_to_update_dict[dict_key] = time_to_update
        epochs_dict[dict_key] = epochs
        print("="*50)

    #====Q6b====
 
    init_dict_char_rnn = init_dict_char_rnn_backup.copy()
    init_dict_word_rnn = init_dict_word_rnn_backup.copy()
    init_dict_char_rnn['rnn_layer'] = 2
    #char_rnn
    tf.reset_default_graph()
    print('training char_rnn_with_2_layers')
    train_err,test_acc,time_to_update,epochs = run_rnn_model(**init_dict_char_rnn)
    dict_key = 'char_rnn_rnn_layer_2'
    train_err_dict[dict_key] = train_err
    test_acc_dict[dict_key] = test_acc
    time_to_update_dict[dict_key] = time_to_update
    epochs_dict[dict_key] = epochs
    print("="*50)

    init_dict_word_rnn['rnn_layer'] = 2
    #word_rnn
    tf.reset_default_graph()
    print('training word_rnn_with_2_layers')
    train_err,test_acc,time_to_update,epochs = run_rnn_model(**init_dict_word_rnn)
    dict_key = 'word_rnn_rnn_layer_2'
    train_err_dict[dict_key] = train_err
    test_acc_dict[dict_key] = test_acc
    time_to_update_dict[dict_key] = time_to_update
    epochs_dict[dict_key] = epochs
    print("="*50)

    #====Q6c====
    init_dict_char_rnn = init_dict_char_rnn_backup.copy()
    init_dict_word_rnn = init_dict_word_rnn_backup.copy()
    init_dict_char_rnn['gradient_clipped']=True
    init_dict_word_rnn['gradient_clipped']=True

    #char_rnn
    tf.reset_default_graph()
    print('training char_rnn_with_gradient_clipped')
    train_err,test_acc,time_to_update,epochs = run_rnn_model(**init_dict_char_rnn)
    dict_key = 'char_rnn_gradient_clipped'
    train_err_dict[dict_key] = train_err
    test_acc_dict[dict_key] = test_acc
    time_to_update_dict[dict_key] = time_to_update
    epochs_dict[dict_key] = epochs
    print("="*50)

    #word_rnn
    tf.reset_default_graph()
    print('training word_rnn_with_gradient_clipped')
    train_err,test_acc,time_to_update,epochs = run_rnn_model(**init_dict_word_rnn)
    dict_key = 'word_rnn_gradient_clipped'
    train_err_dict[dict_key] = train_err
    test_acc_dict[dict_key] = test_acc
    time_to_update_dict[dict_key] = time_to_update
    epochs_dict[dict_key] = epochs
    print("="*50)

    df = pd.DataFrame(columns=['dict_key','train_err_list','test_acc_dict','time_to_update_list','epochs_list','converge_test_accuracy'])
    i = 0
    for dict_key in train_err_dict.keys():
        update_value = [dict_key,
                        train_err_dict[dict_key],
                        test_acc_dict[dict_key],
                        time_to_update_dict[dict_key],
                        epochs_dict[dict_key],
                        test_acc_dict[dict_key][-1]
                        ]
        df.loc[i] = update_value
        i += 1
    df.to_csv('model_results.csv',index=False)


if __name__ == '__main__':
    main()
