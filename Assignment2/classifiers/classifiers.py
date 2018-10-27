__all__ = ['CNNClassifer', 'CNNCharClassifer']

import math
import tensorflow as tf
import numpy as np
import time

seed = 10
tf.set_random_seed(seed)


class CNNClassifer():
    def __init__(
        self,
        input_width=32, input_height=32, num_channels=3, output_dim=10, drop_out=False,
        keep_prob=0.9, hidden_layer_dict=None, num_feature_maps=50,
        batch_size=128, learning_rate=0.001,
        l2_beta=0.001, epochs=1000,
        early_stop=False, patience=20, min_delta=0.001,
        optimizer='GD', momentum=None,
        **kwargs
    ):

        self.input_width = input_width
        self.input_height = input_height
        self.num_channels = num_channels
        self.output_dim = output_dim
        self.drop_out = drop_out
        if self.drop_out:
            self._keep_prob = tf.placeholder(tf.float32)
            self.keep_prob = keep_prob
        self.hidden_layer_dict = hidden_layer_dict
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_beta = l2_beta
        self.epochs = epochs
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        self.optimizer = optimizer
        if momentum:
            self.momentum = momentum
        self._build_model()
    #end def


    def _build_layer(self, x, cwindow_width, cwindow_height, input_maps, output_maps, cstrides, cpadding,
                    pwindow_width, pwindow_height, pstrides, ppadding, **kwargs):
        #Conv
        W = tf.Variable(tf.truncated_normal([cwindow_width, cwindow_height, input_maps, output_maps],
                        stddev=1.0/np.sqrt(input_maps*cwindow_width*cwindow_height)), name='weights')  # [window_width, window_height, input_maps, output_maps]
        b = tf.Variable(tf.zeros([output_maps]), name='biases')

        conv = tf.nn.relu(tf.nn.conv2d(x, W, [1, cstrides, cstrides, 1], padding=cpadding) + b)  # strides = [1, stride, stride, 1]
        
        #Pool
        pool = tf.nn.max_pool(conv, ksize= [1, pwindow_width, pwindow_height, 1], strides= [1, pstrides, pstrides, 1], padding=ppadding, name='pool')

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
        self.Wf = tf.Variable(tf.truncated_normal([dim, self.hidden_layer_dict['F1']['size']], stddev=1.0/np.sqrt(dim)), name='weights')
        self.bf = tf.Variable(tf.zeros([self.hidden_layer_dict['F1']['size']]), name='biases')
        self.hf = tf.nn.relu(tf.matmul(h_pool2_flat, self.Wf) + self.bf)
        
        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        if self.drop_out:
            self.hf = tf.nn.dropout(self.hf, self._keep_prob)

        self.W_output = tf.Variable(tf.truncated_normal([self.hidden_layer_dict['F1']['size'], self.output_dim], stddev=1.0/np.sqrt(self.hidden_layer_dict['F1']['size'])), name='weights')
        self.b_output = tf.Variable(tf.zeros([self.output_dim]), name='biases')

        self.y_conv = tf.matmul(self.hf, self.W_output) + self.b_output
        
        # accuracy
        self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1)), tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        
        # diff choices of optimizer
        if self.optimizer == 'GD': self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
        elif self.optimizer == 'momentum': self.train_op = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cross_entropy)
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

                if i % 100 == 0:
                    print('iter: %d, train error  : %g'%(i, self.train_err[i]))
                    print('iter: %d, test accuracy  : %g'%(i, self.test_acc[i]))
                    print('-'*50)

            #end for
            self.early_stop_epoch = _epochs
            self.saver.save(sess, ".ckpt/1amodel.ckpt")
        #end with

        self.time_taken_one_epoch = (time_to_update/_epochs) * 1000
        return self
    #end def


    def get_feature_maps(self, X):
        with tf.Session() as sess:
            self.saver.restore(sess, ".ckpt/1amodel.ckpt")
            if self.drop_out:
                c1, p1, c2, p2 = sess.run([self.h_conv1, self.h_pool1, self.h_conv2, self.h_pool2], {self.x: X.reshape(-1, self.input_width*self.input_height*self.num_channels), self._keep_prob: 1.0})
            else:
                c1, p1, c2, p2 = sess.run([self.h_conv1, self.h_pool1, self.h_conv2, self.h_pool2], {self.x: X.reshape(-1, self.input_width*self.input_height*self.num_channels)})

        return c1, p1, c2, p2
#end class


class CNNCharClassifer():
    def __init__(
        self,
        input_dim=100, output_dim=10, drop_out=False,
        keep_prob=0.9, hidden_layer_dict=None,
        batch_size=128, learning_rate=0.001,
        l2_beta=0.001, epochs=1000,
        early_stop=False, patience=20, min_delta=0.001,
        optimizer='GD', momentum=None,
        embed=False, vocab_size=256, embed_dim=20,
        **kwargs
    ):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_out = drop_out
        if self.drop_out:
            self._keep_prob = tf.placeholder(tf.float32)
            self.keep_prob = keep_prob
        self.hidden_layer_dict = hidden_layer_dict
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_beta = l2_beta
        self.epochs = epochs
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        self.optimizer = 'GD'
        if momentum:
            self.momentum = momentum
        self.embed = embed
        if self.embed:
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
        self._build_model()
    #end def


    def _build_layer(self, x, cfilters, ckernel_size, cpadding,
                    pwindow, pstrides, ppadding, **kwargs):
        #Conv
        conv = tf.layers.conv2d(
                            x,
                            filters=cfilters,
                            kernel_size=ckernel_size,
                            padding=cpadding,
                            use_bias=True,
                            kernel_initializer=tf.initializers.truncated_normal(seed=seed),
                            bias_initializer=tf.zeros_initializer(),
                            activation=tf.nn.relu)
        
        #Pool
        pool = tf.layers.max_pooling2d(
                            conv,
                            pool_size=pwindow,
                            strides=pstrides,
                            padding=ppadding)

        return conv, pool
    #end def


    def _build_model(self):
        self.x = tf.placeholder(tf.int64, [None, self.input_dim])
        self.y_ = tf.placeholder(tf.int64)
        self.x_ = tf.reshape(tf.one_hot(self.x, 256), [-1, self.input_dim, 256, 1])

        # embed
        if self.embed:
            self.x_embed = tf.contrib.layers.embed_sequence(self.x_, vocab_size=self.vocab_size, embed_dim=self.embed_dim)

        # Conv 1 and pool 1
        input_dict1 = dict(
                        cfilters=self.hidden_layer_dict['C1']['filters'],
                        ckernel_size=self.hidden_layer_dict['C1']['kernel_size'],
                        cpadding=self.hidden_layer_dict['C1']['padding'],
                        pwindow=self.hidden_layer_dict['S1']['window'],
                        pstrides=self.hidden_layer_dict['S1']['strides'],
                        ppadding=self.hidden_layer_dict['S1']['padding'])
        if self.embed:
            self.h_conv1, self.h_pool1 = self._build_layer(self.x_embed, **input_dict1)
        else:
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

        self.h_pool2 = tf.squeeze(tf.reduce_max(self.h_conv2, 1), squeeze_dims=[1])
        self.y_conv = tf.layers.dense(self.h_pool2, self.output_dim, activation=tf.nn.relu)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.y_, self.output_dim), logits=self.y_conv))

        # accuracy
        self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(tf.one_hot(self.y_, self.output_dim), 1)), tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)
        
        # diff choices of optimizer
        if self.optimizer == 'GD': self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
        elif self.optimizer == 'momentum': self.train_op = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cross_entropy)
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

                if i % 100 == 0:
                    print('iter: %d, train error  : %g'%(i, self.train_err[i]))
                    print('iter: %d, test accuracy  : %g'%(i, self.test_acc[i]))
                    print('-'*50)

            #end for
            self.early_stop_epoch = _epochs
            self.saver.save(sess, ".ckpt/1amodel.ckpt")
        #end with

        self.time_taken_one_epoch = (time_to_update/_epochs) * 1000
        return self
    #end def


    def get_feature_maps(self, X):
        with tf.Session() as sess:
            self.saver.restore(sess, ".ckpt/1amodel.ckpt")
            if self.drop_out:
                c1, p1, c2, p2 = sess.run([self.h_conv1, self.h_pool1, self.h_conv2, self.h_pool2], {self.x: X.reshape(-1, self.input_width*self.input_height*self.num_channels), self._keep_prob: 1.0})
            else:
                c1, p1, c2, p2 = sess.run([self.h_conv1, self.h_pool1, self.h_conv2, self.h_pool2], {self.x: X.reshape(-1, self.input_width*self.input_height*self.num_channels)})

        return c1, p1, c2, p2
#end class


