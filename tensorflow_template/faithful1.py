#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 19:37:14 2016

@author: yuntao
"""

import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import math
import numpy as np

faithful_dataframe = pd.read_csv('faithful.csv',sep = ',', header = 0)
faithful_values = faithful_dataframe.values[:,:-1]

train_idx = batch_size = int(math.floor(len(faithful_values)*0.8))
batch_x = faithful_values[:train_idx,:]
batch_y = faithful_values[1:train_idx+1,:]

learning_rate = 0.001
training_iters = 10000000
display_step = 10

# Network Parameters
n_input = 2 # input size
n_steps = 16 # timesteps
n_hidden = 16 # hidden layer num of features
n_classes = 1 # MNIST total classes (0-9 digits)

i = 0
x_input = np.array([])
y_input = np.array([])
while i+n_steps <= train_idx:
    x_input = np.append(x_input, faithful_values[i:i+n_steps,:])
    y_input = np.append(y_input, faithful_values[i+1,0])
    i+=1
x_input = x_input.reshape(-1,n_steps,2)
y_input = y_input.reshape(-1,1)

x_test = np.array([])
y_test = np.array([])
while i+n_steps < len(faithful_values):
    x_test = np.append(x_test, faithful_values[i:i+n_steps,:])
    y_test = np.append(y_test, faithful_values[i+1,0])
    i+=1
x_test = x_test.reshape(-1,n_steps,2)
y_test = y_test.reshape(-1,1)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Permuting batch_size and n_steps
    # x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0,state_is_tuple = True)
    lstm_cell = rnn_cell.MultiRNNCell([lstm_cell]*2,state_is_tuple = True)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean((pred-y)**2)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = x_input, y_input
        # Reshape data to get 28 seq of 28 elements
        #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            test_loss = sess.run(cost, feed_dict={x: x_test, y: y_test})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) +",Test Loss = " +  "{:.6f}".format(test_loss))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
#    test_len = 128
#    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
#    test_label = mnist.test.labels[:test_len]