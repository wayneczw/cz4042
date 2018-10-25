import numpy as np
import pandas
import tensorflow as tf
import csv
from classifiers import CNNCharClassifer


MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15

no_epochs = 100
learning_rate = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def char_cnn_model(x):
  
    input_layer = tf.reshape(
        tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

    with tf.variable_scope('CNN_Layer1'):
        conv1 = tf.layers.conv2d(
                            input_layer,
                            filters=N_FILTERS,
                            kernel_size=FILTER_SHAPE1,
                            padding='VALID',
                            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
                            conv1,
                            pool_size=POOLING_WINDOW,
                            strides=POOLING_STRIDE,
                            padding='SAME')

        pool1 = tf.squeeze(tf.reduce_max(conv1, 1), squeeze_dims=[1])

        logits = tf.layers.dense(pool1, MAX_LABEL, activation=None)

    return input_layer, logits


def read_data_chars():
  
    x_train, y_train, x_test, y_test = [], [], [], []

    with open('data/train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[1])
            y_train.append(int(row[0]))

    with open('data/test_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[1])
            y_test.append(int(row[0]))
      
    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)
      
      
    char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
    x_train = np.array(list(char_processor.fit_transform(x_train)))
    x_test = np.array(list(char_processor.transform(x_test)))
    y_train = y_train.values
    y_test = y_test.values
      
    return x_train, y_train, x_test, y_test

  
def main():
  
    trainX, trainY, testX, testY = read_data_chars()
    valX = testX
    valY = testY
    trainX = trainX[:100]
    trainY = trainY[:100]

    C1_dict = dict(filters=10, kernel_size=[20, 256], padding='VALID')
    C2_dict = dict(filters=10, kernel_size=[20, 1], padding='VALID')
    S1_dict = dict(window=4, padding='SAME', strides=2)
    S2_dict = dict(window=4, padding='SAME', strides=2)
    hidden_layer_dict = dict(C1=C1_dict, C2=C2_dict, S1=S1_dict, S2=S2_dict)

    init_dict = dict(input_dim=MAX_DOCUMENT_LENGTH, output_dim=MAX_LABEL,
                hidden_layer_dict=hidden_layer_dict,
                batch_size=128, learning_rate=learning_rate, epochs=1000,
                early_stop=True, patience=20, min_delta=0.001, optimizer='Adam')
    cnn = CNNCharClassifer(**init_dict).train(X_train=trainX, Y_train=trainY,
                                        X_test=testX, Y_test=testY,
                                        X_val=valX, Y_val=valY)
    train_err, test_acc, time_taken_one_epoch, early_stop_epoch = cnn.train_err, cnn.test_acc, cnn.time_taken_one_epoch, cnn.early_stop_epoch
#end def


if __name__ == '__main__':
  main()
