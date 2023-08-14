#!/usr/bin/env python
# coding: utf-8

# #### Version 2 of Sparisity Comparison
# This program copies the RadiXNet implementation from RadixNet.ipynb, but instead uses it to create multiple networks with differing sparsities to compare the difference in accuracies between sparsities
# Is much neater and more specified to investigating the models with sparsity >0.9 that have irregular training

# ## Imports

# In[2]:


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_model_optimization as tfmot
import RadixNetFuncs
import matlab.engine
import math
import networkx as nx
from ast import literal_eval


# ### Read data

# In[3]:


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


# # Create different models based on sparsity

# In[3]:


# desired_network_structure = [784, 300, 100, 10]
# factors = list(range(1,11))
# permutations = RadixNetFuncs.generate_permutations(desired_network_structure,factors)
# print(len(permutations))


# In[ ]:


# topologies = RadixNetFuncs.build_radix_nets(permutations,desired_network_structure)


# In[4]:


# desired = [784,300,100,10]
# sparsemodels, greater90 = RadixNetFuncs.findSimpleModels(desired)
# print(len(greater90))


# In[5]:


# myKeys = list(greater90.keys())
# myKeys.sort()
# #print(myKeys)
# models = []
# sampleSize = 400
# sparsities = []
# rstructures = []
# for i in range(0,len(myKeys),len(myKeys)//sampleSize):
#     sparsity = myKeys[i]
#     sparsities.append(sparsity)
#     rstructures.append(greater90[sparsity])
#     model = RadixNetFuncs.createModel(greater90[sparsity])
#     model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     models.append(model)
# for i in range(len(sparsities)):
#     print(str(sparsities[i])+": "+str(rstructures[i]))


# In[7]:


# histories = []

# start = 408
# for i in range(start,len(models)):
#     model = models[i]
#     #print("model " +str(i+1)+" out of "+str(len(models)))
#     epochs = 3
#     history = model.fit(X_train,Y_train,epochs = epochs, batch_size = 100,validation_split = 0.1, verbose = 0)
#     acc = history.history['accuracy']
#     if acc[-1]<=0.75:
#         file = open("smodels.txt", "a")
#         file.write(str(i)+" - ")
#         file.write(str(sparsities[i])+": "+str(rstructures[i])+", "+str(acc[-1])+"\n")
#         file.close()
#     histories.append(history)
models = []
sparsities = []
structures = []
file = open("smodels.txt", "r")
lines = file.readlines()
file.close()
desired = [784,300,100,10]
for line in lines:
    line = line.split(" - ")
    sparsities.append(float(line[1]))
    structures.append(literal_eval(line[2]))
    structure = RadixNetFuncs.build_radix_nets([structures[-1][0]], desired)
    #print(structure[0].shape)
    model = RadixNetFuncs.CustomModel(structure)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    models.append(model)

epochs = 10
histories = []
for i, model in enumerate(models):
    print("model " +str(i+1)+" out of "+str(len(models)))
    history = model.fit(X_train,Y_train,epochs = epochs, batch_size = 512,validation_split = 0.1, verbose = 0)
    histories.append(history)

epochs_range = range(epochs)
plt.figure(figsize=(10, 10))
for i, history in enumerate(histories):
    acc = history.history['accuracy']
    
    plt.plot(epochs_range, acc)
plt.legend(loc='lower right')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
    
plt.savefig('Artifacts/all.png')

# In[ ]:




