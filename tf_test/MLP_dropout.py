# adapted from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import math
import numpy as np

print('program starting ...')

desired_dataframe = pd.read_csv('desired.csv',sep = ',', header = None)
desired_values = desired_dataframe.values[:,0]
desired_values = np.transpose(desired_values.reshape((5,-1)))

input_dataframe = pd.read_csv('input.csv',sep = ',',header = None)
input_values = np.array(input_dataframe.values[:,0],dtype = 'float')
input_values = input_values.reshape((5,300,-1))

def standardize_set(arr):
    a0 = np.multiply(np.add(arr[0,:],-125),0.01)
    a1 = np.multiply(np.add(arr[1,:],-125),0.01)
    a2 = np.multiply(np.add(arr[2,:],-125),0.01)
    a3 = np.multiply(np.add(arr[3,:],-250),0.01)
    a4 = np.multiply(np.add(arr[4,:],-250),0.01)
    return np.array([a0,a1,a2,a3,a4])


five, threehundred, total_size = input_values.shape
for j in range(total_size):
    input_values[:,:,j] = standardize_set(input_values[:,:,j])

train_idx = batch_size = int(math.floor(total_size*0.8))
test_size = total_size - batch_size;
train_x = input_values[:,:,:train_idx]
train_d = desired_values[:train_idx,:]
test_x = input_values[:,:,train_idx:]
test_d = desired_values[train_idx:,:]

tr_x = np.empty([600,batch_size])
for i in range(batch_size):
    tr_x[:,i] = np.concatenate([train_x[3,:,i],train_x[4,:,i]])
tst_x = np.empty([600,test_size])
for i in range(test_size):
    tst_x[:,i] = np.concatenate([test_x[3,:,i],test_x[4,:,i]])
tr_x = np.transpose(tr_x)
tst_x = np.transpose(tst_x)
#now the input are of size [batch_size, n_input]

learning_rate = 0.005
training_iters = 1000000
display_step = 5

# Network Parameters
n_input = 600 # input size, currently only using col 3,4 (4,5 in matlab)
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_classes = 5 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder("float")
# Create model
def multilayer_perceptron(x, weights, biases,keep_prob):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1_drop = tf.nn.dropout(layer_1,keep_prob) 
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1_drop, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2_drop = tf.nn.dropout(layer_2,keep_prob)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2_drop, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = multilayer_perceptron(x, weights, biases,keep_prob)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
print("prep finished, starting ...")
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = tr_x, train_d
        # Reshape data to get 28 seq of 28 elements
        #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob:0.5})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
            # Calculate batch loss
            # loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            test_accuracy = sess.run(accuracy, feed_dict={x: tst_x, y: test_d, keep_prob:1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Accuracy= " + \
                  "{:.6f}".format(acc) +",Test Accuracy = " +  "{:.6f}".format(test_accuracy))
        step += 1
    print("Optimization Finished!")
