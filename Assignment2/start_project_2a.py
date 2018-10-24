#
# Project 2, starter code Part a
#

import math
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import pickle
from sklearn.model_selection import train_test_split
from classifiers import CNNClassifer
import time

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
    error_against_epoch = [0, 1000, 0, 2]
    accuracy_against_epoch = [0, 1000, 0, 1]

    trainX, trainY, valX, valY, testX, testY = load_data('data/data_batch_1', 'data/test_batch_trim')
    trainX = trainX[:200]
    trainY = trainY[:200]
    valX = valX[:10]
    valY = valY[:10]
    testX = testX[:10]
    testY = testY[:10]

    ################ Q1
    C1_dict = dict(window_width=9, window_height=9, output_maps=50, padding='VALID', strides=1)
    C2_dict = dict(window_width=5, window_height=5, output_maps=60, padding='VALID', strides=1)
    S1_dict = dict(window_width=2, window_height=2, padding='VALID', strides=2)
    S2_dict = dict(window_width=2, window_height=2, padding='VALID', strides=2)
    F1_dict = dict(size=300)
    hidden_layer_dict = dict(C1=C1_dict, C2=C2_dict, S1=S1_dict, S2=S2_dict, F1=F1_dict)

    init_dict = dict(input_width=IMG_SIZE, input_height=IMG_SIZE, num_channels=NUM_CHANNELS, output_dim=NUM_CLASSES,
                hidden_layer_dict=hidden_layer_dict, num_feature_maps=50,
                batch_size=128, learning_rate=learning_rate, epochs=1000,
                early_stop=True, patience=20, min_delta=0.001)
    cnn = CNNClassifer(**init_dict).train(X_train=trainX, Y_train=trainY,
                                        X_test=testX, Y_test=testY,
                                        X_val=valX, Y_val=valY)
    train_err, test_acc, time_taken_one_epoch, early_stop_epoch = cnn.train_err, cnn.test_acc, cnn.time_taken_one_epoch, cnn.early_stop_epoch

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

    ind = np.random.randint(low=0, high=10)
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


    ################ Q2

    ################ Q3
    C1_dict = dict(window_width=9, window_height=9, output_maps=50, padding='VALID', strides=1)
    C2_dict = dict(window_width=5, window_height=5, output_maps=60, padding='VALID', strides=1)
    S1_dict = dict(window_width=2, window_height=2, padding='VALID', strides=2)
    S2_dict = dict(window_width=2, window_height=2, padding='VALID', strides=2)
    F1_dict = dict(size=300)
    hidden_layer_dict = dict(C1=C1_dict, C2=C2_dict, S1=S1_dict, S2=S2_dict, F1=F1_dict)

    train_err_dict = dict()
    test_acc_dict = dict()
    time_taken_one_epoch_dict = dict()
    early_stop_epoch_dict = dict()

    optimizers = [dict(optimizer='momentum', momentum=0.1),
                dict(optimizer='RMSProp'),
                dict(optimizer='Adam'),
                dict(drop_out=True, keep_prob=0.9)]
    #### momentum = 0.1
    for optimizer in optimizers:
        print('='*50)
        print(optimizer.values())
        init_dict = dict(input_width=IMG_SIZE, input_height=IMG_SIZE, num_channels=NUM_CHANNELS, output_dim=NUM_CLASSES, 
            hidden_layer_dict=hidden_layer_dict, num_feature_maps=50,
            batch_size=128, learning_rate=learning_rate, epochs=1000,
            early_stop=True, patience=20, min_delta=0.001, **optimizer)
        cnn = CNNClassifer(**init_dict).train(X_train=trainX, Y_train=trainY,
                                            X_test=testX, Y_test=testY,
                                            X_val=valX, Y_val=valY)
        train_err, test_acc, time_taken_one_epoch, early_stop_epoch = cnn.train_err, cnn.test_acc, cnn.time_taken_one_epoch, cnn.early_stop_epoch
        try:
            train_err_dict[optimizer['optimizer']] = train_err
            test_acc_dict[optimizer['optimizer']] = test_acc
            time_taken_one_epoch_dict[optimizer['optimizer']] = time_taken_one_epoch
            early_stop_epoch_dict[optimizer['optimizer']] = early_stop_epoch
        except KeyError:
            train_err_dict['Drop Out 0.9'] = train_err
            test_acc_dict['Drop Out 0.9'] = test_acc
            time_taken_one_epoch_dict['Drop Out 0.9'] = time_taken_one_epoch
            early_stop_epoch_dict['Drop Out 0.9'] = early_stop_epoch
        #end try
    #end for

    # Plot Training Errors
    plt.figure("Train Error against Epoch")
    plt.title("Train Error against Epoch")
    error_against_epoch[1] = max([len(l) for l in train_err_dict.values()])
    plt.axis(error_against_epoch)
    for key, val in train_err_dict.items():
        plt.plot(range(len(val)), val, label = 'optimizer = {}'.format(key))
        plt.xlabel('Epochs')
        plt.ylabel('Train Error')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.savefig('figures/a/3_train_error_vs_epoch.png')

    # Plot Test Accuracy
    plt.figure("Test Accuracy against Epoch")
    plt.title("Test Accuracy against Epoch")
    accuracy_against_epoch[1] = max([len(l) for l in test_acc_dict.values()])
    plt.axis(accuracy_against_epoch)
    for key, val in test_acc_dict.items():
        plt.plot(range(len(val)), val, label = 'optimizer = {}'.format(key))
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(b=True)
    #end for
    plt.savefig('figures/a/3_test_accuracy_vs_epoch.png')
    #end for

    # Plot Time Taken for One Epoch
    optimizer_list = test_acc_dict.keys()
    time_taken_one_epoch_list = [time_taken_one_epoch_dict[optimizer] for optimizer in optimizer_list]
    plt.figure("Time Taken for One Epoch")
    plt.title("Time Taken for One Epoch")
    plt.xticks(np.arange(len(optimizer_list)), optimizer_list)
    plt.plot(optimizer_list, time_taken_one_epoch_list)
    plt.xlabel('Optimizer')
    plt.ylabel('Time per Epoch/ms')
    plt.grid(b=True)
    plt.savefig('figures/a/3_time_taken_for_one_epoch.png')

    early_stop_epoch_list = [early_stop_epoch_dict[optimizer] for optimizer in optimizer_list]
    total_time_taken_list = [x*y for x,y in zip(early_stop_epoch_list,time_taken_one_epoch_list)]
    # Plot Total Time Taken
    plt.figure("Early Stopping Total Time Taken")
    plt.title("Early Stopping Total Time Taken")
    plt.plot(optimizer_list, total_time_taken_list)
    plt.xlabel('Optimizer')
    plt.ylabel('Total Time/ms')
    plt.grid(b=True)
    plt.savefig('figures/a/3_total_time_taken.png')



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
   

if __name__ == '__main__':
  main()
