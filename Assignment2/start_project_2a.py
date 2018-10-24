#
# Project 2, starter code Part a
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
from sklearn.model_selection import train_test_split
from classifiers import CNNClassifer


NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 10
batch_size = 128


seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

def load_data(train_file, test_file):
    with open(train_file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')
    data, label = samples['data'], samples['labels']
    train_data, val_data, train_labels, val_labels = train_test_split(data, label, test_size=0.25)
    
    # train
    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)  
    train_labels_ = np.zeros([train_labels.shape[0], NUM_CLASSES])
    train_labels_[np.arange(train_labels.shape[0]), train_labels-1] = 1

    # validation
    val_data = np.array(val_data, dtype=np.float32)
    val_labels = np.array(val_labels, dtype=np.int32)  
    val_labels_ = np.zeros([val_labels.shape[0], NUM_CLASSES])
    val_labels_[np.arange(val_labels.shape[0]), val_labels-1] = 1

    # test
    with open(test_file, 'rb') as fo:
        try:
            test_samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            test_samples = pickle.load(fo, encoding='latin1')
    test_data, test_labels = test_samples['data'], test_samples['labels']
    test_data = np.array(test_data, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.int32)  
    test_labels_ = np.zeros([test_labels.shape[0], NUM_CLASSES])
    test_labels_[np.arange(test_labels.shape[0]), test_labels-1] = 1

    # scale
    train_min = np.min(train_data, axis = 0)
    train_max = np.max(train_data, axis = 0)
    train_data = (train_data - train_min)/train_max
    val_data = (val_data - train_min)/train_max
    test_data = (test_data - train_min)/train_max

    return train_data, train_labels_, val_data, val_labels_, test_data, test_labels_
#end def


def main():

    trainX, trainY, valX, valY, testX, testY = load_data('data/data_batch_1', 'data/test_batch_trim')
    trainX = trainX[:1000]
    trainY = trainY[:1000]
    valX = valX[:100]
    valY = valY[:100]
    testX = testX[:100]
    testY = testY[:100]

    C1_dict = dict(window_width=9, window_height=9, output_maps=50, padding='VALID', strides=1)
    C2_dict = dict(window_width=5, window_height=5, output_maps=60, padding='VALID', strides=1)
    S1_dict = dict(window_width=2, window_height=2, padding='VALID', strides=2)
    S2_dict = dict(window_width=2, window_height=2, padding='VALID', strides=2)
    F1_dict = dict(size=300)
    hidden_layer_dict = dict(C1=C1_dict, C2=C2_dict, S1=S1_dict, S2=S2_dict, F1=F1_dict)
    cnn = CNNClassifer(hidden_layer_dict=hidden_layer_dict).train(X_train=trainX, Y_train=trainY,
                                                                X_test=testX, Y_test=testY,
                                                                X_val=valX, Y_val=valY)
    print(cnn.test_acc)

    # ind = np.random.randint(low=0, high=100)
    # X = trainX[ind,:]
    
    # plt.figure()
    # plt.gray()
    # X_show = X.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
    # plt.axis('off')
    # plt.imshow(X_show)
    # plt.savefig('./p1b_2.png')


if __name__ == '__main__':
  main()
