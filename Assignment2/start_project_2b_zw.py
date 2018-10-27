import numpy as np
import pandas
import tensorflow as tf
import csv
from sklearn.model_selection import train_test_split

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15

no_epochs = 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)


class CNNClassifer():
    def __init__(
        self,
        input_dim=100, output_dim=15, drop_out=False,
        drop_out_rate=0.5, hidden_layer_dict=None,
        batch_size=128, learning_rate=0.01,
        epochs=1000,
        early_stop=False, patience=20, min_delta=0.001,
        choice='char',
        **kwargs
    ):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_out = drop_out
        if self.drop_out:
            self.drop_out_rate = drop_out_rate
        self.hidden_layer_dict = hidden_layer_dict
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        if choice == 'char': self._build_char_model()
        else: self._build_word_model()
    #end def


    def _build_layer(self, x, cfilters, ckernel_size, cpadding,
                    pwindow, pstrides, ppadding, **kwargs):
        # Conv
        conv = tf.layers.conv2d(
                            x,
                            filters=cfilters,
                            kernel_size=ckernel_size,
                            padding=cpadding,
                            use_bias=True,
                            kernel_initializer=tf.initializers.truncated_normal(seed=seed),
                            bias_initializer=tf.zeros_initializer(),
                            activation=tf.nn.relu)

        # dropout
        if self.drop_out:
            conv = tf.layers.dropout(
                                conv,
                                rate=self.drop_out_rate,
                                seed=seed)
        
        # Pool
        pool = tf.layers.max_pooling2d(
                            conv,
                            pool_size=pwindow,
                            strides=pstrides,
                            padding=ppadding)

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

        self.h_pool2 = tf.squeeze(tf.reduce_max(self.h_conv2, 1), squeeze_dims=[1])
        self.y_conv = tf.layers.dense(self.h_pool2, self.output_dim, activation=tf.nn.relu)

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
                                                x,
                                                vocab_size=n_words,
                                                embed_dim=EMBEDDING_SIZE)
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

        self.h_pool2 = tf.squeeze(tf.reduce_max(self.h_conv2, 1), squeeze_dims=[1])
        self.y_conv = tf.layers.dense(self.h_pool2, self.output_dim, activation=tf.nn.relu)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.y_, self.output_dim), logits=self.y_conv))

        # accuracy
        self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(tf.one_hot(self.y_, self.output_dim), 1)), tf.float32)
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
#end class


def read_data(choice='char'):
    if choice == 'char': row_num = 1
    else: row_num = 2

    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[row_num])
            y_train.append(int(row[0]))
        #end for
    #end with

    with open('test_medium.csv', encoding='utf-8') as filex:
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

    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_val = pandas.Series(x_val)
    y_val = pandas.Series(y_val)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values   
    
    if choice == 'char':
        char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
        x_train = np.array(list(char_processor.fit_transform(x_train)))
        x_val = np.array(list(char_processor.transform(x_val)))
        x_test = np.array(list(char_processor.transform(x_test)))
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
        x_train = np.array(list(vocab_processor.fit_transform(x_train)))
        x_val = np.array(list(vocab_processor.transform(x_val)))
        x_test = np.array(list(vocab_processor.transform(x_test)))

        no_words = len(vocab_processor.vocabulary_)
        print('Total words: %d' % no_words)

        return x_train, y_train, x_val, y_val, x_test, y_test, no_words
    #end if
#end def


def arg_dict(model_save_path, C1_map=50, C2_map=60, optimizer='GD', drop_out=False, keep_prob=0.9):
            input_dict2 = dict(
                        cfilters=self.hidden_layer_dict['C2']['filters'],
                        ckernel_size=self.hidden_layer_dict['C2']['kernel_size'],
                        cpadding=self.hidden_layer_dict['C2']['padding'],
                        pwindow=self.hidden_layer_dict['S2']['window'],
                        pstrides=self.hidden_layer_dict['S2']['strides'],
                        ppadding=self.hidden_layer_dict['S2']['padding'])

    C1_dict = dict(filters=9, kernel_size=9, padding='VALID')
    C2_dict = dict(filters=9, kernel_size=9, padding='VALID')
    S1_dict = dict(window=2, strides=2, padding='VALID')
    S2_dict = dict(window=2, strides=2, padding='VALID')
    hidden_layer_dict = dict(C1=C1_dict, C2=C2_dict, S1=S1_dict, S2=S2_dict)

    init_dict = dict(save_path=model_save_path,optimizer=optimizer,
        input_width=IMG_SIZE, input_height=IMG_SIZE, num_channels=NUM_CHANNELS, output_dim=NUM_CLASSES,
        hidden_layer_dict=hidden_layer_dict,
        batch_size=128, learning_rate=learning_rate, epochs=1000,
        early_stop=True, patience=20, min_delta=0.001,
        drop_out=drop_out, keep_prob=keep_prob)
    return init_dict
#end def


def main():
  
    x_train, y_train, x_val, y_val, x_test, y_test = read_data(choice='char')

    # =========================== Q1

    char_cnn = CNNClassifer().train(
                                X_train=x_train,
                                Y_train=y_train,
                                X_test=x_test,
                                Y_test=y_test,
                                X_val=x_val,
                                Y_val=y_val)

if __name__ == '__main__':
  main()
