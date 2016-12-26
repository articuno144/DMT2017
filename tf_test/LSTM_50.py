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
train_x = input_values[:,:,:train_idx]
train_d = desired_values[:train_idx,:]
test_x = input_values[:,:,train_idx:]
test_d = desired_values[train_idx:,:]

train_x = np.transpose(train_x,(2,1,0))
test_x = np.transpose(test_x,(2,1,0))

learning_rate = 0.005
training_iters = 1000000
display_step = 5

# Network Parameters
n_input = 5 # input size, currently only using col 3,4 (4,5 in matlab)
n_steps = 300 # timesteps
n_hidden = 16 # hidden layer num of features
n_classes = 5 # MNIST total classes (0-9 digits)

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
    # Example data input shape: (batch_size, n_steps, n_input)
    # Current data input shape: (n_input, n_steps, batch_size)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1,0,2])
    # Now the shape is (n_steps, batch_size, n_input)
    # Transpose to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0,state_is_tuple = True)
    lstm_cell = rnn_cell.MultiRNNCell([lstm_cell]*4,state_is_tuple = True)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
print("prep finished, starting ...")
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = train_x, train_d
        # Reshape data to get 28 seq of 28 elements
        #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y: test_d})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) +",Test Accuracy = " +  "{:.6f}".format(test_accuracy))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
#    test_len = 128
#    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
#    test_label = mnist.test.labels[:test_len]
