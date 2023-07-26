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
np.random.seed(100)
np.random.shuffle(data) 

data_train = data.T
Y_train = np.array(data_train[0])
X_train = np.array(data_train[1:]).T
X_train = X_train / 255

test = np.array(test)
data_test = test.T
Y_test = np.array(data_test[0])
X_test = np.array(data_test[1:]).T
X_test = X_test / 255


@pytest.mark.parametrize(
    "layerval,var,exp", 
    [
        (np.array([[1,0],[0,1]]),tf.Variable(tf.constant(np.array([[3,3],[3,3]]),dtype=tf.float32)),np.array([[3,0],[0,3]])),
        (np.array([[1,1,1],[1,1,1],[1,1,1]]),tf.Variable(tf.constant(np.array([[3,3,3],[3,3,3],[3,3,3]]),dtype=tf.float32)),np.array([[3,3,3],[3,3,3],[3,3,3]])),
        (np.array([[0]]),tf.Variable(tf.constant(np.array([[910]]),dtype=tf.float32)),np.array([[0]])),
        (np.array([[1,0,1],[1,0,1]]),tf.Variable(tf.constant(np.array([[3,3,3],[3,3,3]]),dtype=tf.float32)),np.array([[3,0,3],[3,0,3]])),
        (np.array([[1,0,1],[1,0,1]]),tf.Variable(tf.constant(np.array([[3,3,3],[3,3,3]]),dtype=tf.float32)),np.array([[3,0,3],[3,0,3]])),
    ]
)
def test_mask(layerval,var,exp):
    tf.keras.utils.set_random_seed(100)
    tf.config.experimental.enable_op_determinism()
    mask = Mask(layerval)
    masked = tf.math.multiply(var, mask())
    assert np.array_equal(masked.numpy(),exp)


@pytest.mark.parametrize(
    "layerval,input,exp", 
    [
        (np.array([[1,0],[0,0]]),tf.constant(np.array([[5,5],[5,5]]),dtype=tf.float32),np.array([[-0.08136355,0],[-0.08136355,0]])),
        (np.array([[1,0,1],[1,1,0],[0,1,1]]),tf.Variable(tf.constant(np.array([[3,3,3],[3,3,3],[3,3,3]]),dtype=tf.float32)),
         np.array([[ 0.04988966, -0.2344105,   0.15741587],
 [ 0.04988966,-0.2344105,   0.15741587],
 [ 0.04988966, -0.2344105,   0.15741587]])),
        (np.array([[0,0],[0,0]]),tf.constant(np.array([[5,5],[5,5]]),dtype=tf.float32),np.array([[0,0],[0,0]])),
        (np.array([[1,1],[1,1]]),tf.constant(np.array([[5,5],[5,5]]),dtype=tf.float32),np.array([[0.11646298, -0.03761561]
 ,[ 0.11646298, -0.03761561]])),
        (np.array([[5,5],[5,5]]),tf.constant(np.array([[1]]),dtype=tf.float32),np.array([[]]))
    ]
)
def test_Radix(layerval, input, exp):
    tf.keras.utils.set_random_seed(100)
    tf.config.experimental.enable_op_determinism()
    rlayer = RadixLayer(2, layerval)
    if input.shape[1]!=layerval.shape[0]:
        output = rlayer(input)
        assert output==None
        return
    
    output = rlayer(input).numpy()
    if output.shape!=exp.shape:
        assert False, "shapes of RadixLayer and Expected numpy array are not the same"
    print(output)
    output = output.flatten()
    exp = exp.flatten()
    
    for i in range(len(output)):
        if math.isclose(output[i],exp[i],abs_tol = 1e-8)==False:
            assert False, "expected RadixLayer values are different from output"
    
    assert True

@pytest.mark.parametrize(
    "list,exp", 
    [
        ([np.array([[1,0],[0,1]])],0.5),
        ([np.array([[1,0],[0,0]]),np.array([[1,0],[0,0]])],0.75),
        ([np.array([[1,0,0],[0,0,0],[0,0,0]]),np.array([[1,0,0],[0,0,0],[0,0,0]]),np.array([[1,1,1],[1,1,1],[1,1,1]])],0.592592592592),
        ([np.array([[0,0],[0,0]])],1),
        ([np.array([[1,1],[1,1]])],0)
                        
    ]
)
def test_sparsity(list, exp):
    assert math.isclose(calculate_sparsity(list),exp,abs_tol=1e-8)

@pytest.mark.parametrize(
    "N,B,expsparsity,exploss,expacc", 
    [
        ([[10,10], [10]],[8, 3, 1, 1],0.9, 0.32369208335876465, 0.9104910492897034),
        ([[10], [5,2]],[80, 30, 10, 1],0.05935386927122466,0.15198080241680145, 0.9567956924438477), 
        ([[10,10,1]],[8, 3, 1, 1],0.9003380916604057,1.2603896856307983, 0.6031603217124939) ,
        ([[2, 2], [2]], [196, 75, 25, 3], 0.5, 0.16933931410312653, 0.9521952271461487),
        ([[1, 1], [1]], [784, 300, 100, 10], 0,0.1043885126709938, 0.968596875667572)
    ]
)
def test_model(N,B,expsparsity,exploss,expacc):
    tf.keras.utils.set_random_seed(100)
    tf.config.experimental.enable_op_determinism()
    desired = [784,300,100,10]
    rlayers = kemr_net(N,B)
    for i in range(len(rlayers)):
        if rlayers[i].shape[0]>desired[i]:
            rlayers[i] = rlayers[i][:desired[i],:]
        if rlayers[i].shape[1]>desired[i+1]:
            rlayers[i] = rlayers[i][:,:desired[i+1]]
    model = CustomModel(rlayers)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(calculate_sparsity(rlayers))
    assert calculate_sparsity(rlayers) == expsparsity
    epochs = 5
    history = model.fit(X_train,Y_train,epochs = epochs, batch_size = 500,validation_split = 0.1)
    test_loss, test_acc = model.evaluate(X_test,Y_test)
    print(test_loss,test_acc)
    assert math.isclose(test_loss,exploss,abs_tol=1e-8) and math.isclose(test_acc,expacc,abs_tol=1e-8)


#test_model([[1, 1], [1]], [784, 300, 100, 10], 0, 0.1043885126709938, 0.968596875667572)