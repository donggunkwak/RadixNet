from RadixNet import Mask, RadixLayer, CustomModel, calculate_sparsity
import pytest
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_model_optimization as tfmot
from RadixNetCalc import emr_net, kemr_net
import math

data = pd.read_csv("MNIST/mnist_train.csv") #mnist train
test = pd.read_csv("MNIST/mnist_test.csv") #mnist test
data = np.array(data)
test = np.array(test)

def test_mask_1():
    layerval = np.array([[1,0],[0,1]])
    var = tf.Variable(tf.constant(np.array([[3,3],[3,3]]),dtype=tf.float32))
    mask = Mask(layerval)
    masked = tf.math.multiply(var, mask())
    assert np.array_equal(masked.numpy(),np.array([[3,0],[0,3]]))

def test_Radix_1():
    layerval = np.array([[1,0],[0,0]])
    rlayer = RadixLayer(2, layerval)
    input = tf.constant(np.array([[5,5],[5,5]]),dtype=tf.float32)
    output = rlayer(input).numpy()
    print(output)
    assert output[0][1]==output[1][1]==0 and output[0][0]==output[1][0]==1.25

def test_sparsity_1():
    list = [np.array([[1,0],[0,1]])]
    assert calculate_sparsity(list)==0.5

def test_model_1():
    
    data_train = data.T
    Y_train = np.array(data_train[0])
    X_train = np.array(data_train[1:]).T
    X_train = X_train / 255
    
    data_test = test.T
    Y_test = np.array(data_test[0])
    X_test = np.array(data_test[1:]).T
    X_test = X_test / 255
    N = [[10,10], [10]]    
    B = [8, 3, 1, 1]
    desired = [784,300,100,10]
    rlayers = kemr_net(N,B)
    for i in range(len(rlayers)):
        if rlayers[i].shape[0]>desired[i]:
            rlayers[i] = rlayers[i][:desired[i],:]
        if rlayers[i].shape[1]>desired[i+1]:
            rlayers[i] = rlayers[i][:,:desired[i+1]]
    model = CustomModel(rlayers)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    assert calculate_sparsity(rlayers) == 0.9
    epochs = 5
    history = model.fit(X_train,Y_train,epochs = epochs, batch_size = 20,validation_split = 0.1)
    test_loss, test_acc = model.evaluate(X_test,Y_test)
    print(test_loss,test_acc)
    #assert test_loss==0.25218749046325684 and test_acc==0.9305930733680725

test_model_1()


