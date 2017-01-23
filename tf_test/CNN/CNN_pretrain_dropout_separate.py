# adapted from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import math
import numpy as np

print('program starting ...')

desired_dataframe = pd.read_csv('d2.csv',sep = ',', header = None)
desired_values = desired_dataframe.values[:,0]
desired_values = np.transpose(desired_values.reshape((5,-1)))

input_dataframe = pd.read_csv('x2.csv',sep = ',',header = None)
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

train_idx = batch_size = 3071
ptrain_idx = 3191
test_size = total_size - ptrain_idx
train_x = input_values[:,:,:train_idx]
train_d = desired_values[:train_idx,:]
ptrain_x = input_values[:,:,train_idx:ptrain_idx]
ptrain_d = desired_values[train_idx:ptrain_idx,:]
test_x = input_values[:,:,ptrain_idx:]
test_d = desired_values[ptrain_idx:,:]

tr_x = np.empty([300,2,batch_size])
for i in range(batch_size):
    tr_x[:,0,i] = train_x[3,:,i]
    tr_x[:,1,i] = train_x[4,:,i]
ptr_x = np.empty([300,2,120])
for i in range(120):
    ptr_x[:,0,i] = ptrain_x[3,:,i]
    ptr_x[:,1,i] = ptrain_x[4,:,i]
tst_x = np.empty([300,2,test_size])
for i in range(test_size):
    tst_x[:,0,i] = test_x[3,:,i]
    tst_x[:,1,i] = test_x[4,:,i]
tr_x = np.transpose(tr_x,[2,0,1])
ptr_x = np.transpose(ptr_x,[2,0,1])
tst_x = np.transpose(tst_x,[2,0,1])
#now the input are of size [batch_size, n_input,n_sensors]

learning_rate = 0.005
training_iters = 500000
display_step = 5

# Network Parameters
n_input = 300 # input size, currently only using col 3,4 (4,5 in matlab)
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_classes = 5 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input, 2])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder("float")
# Create model
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, n_input,2, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {
    # 5x5 conv, 2 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([3, 2, 1, 64])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 1, 64, 128])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([73*128, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

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
sess = tf.InteractiveSession()
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
        test_accuracy = sess.run(accuracy, feed_dict={x: tst_x, y: test_d, keep_prob:1.0})
        print("Iter " + str(step*batch_size) + ", Minibatch Accuracy= " + \
              "{:.6f}".format(acc) +",Test Accuracy = " +  "{:.6f}".format(test_accuracy))
    step += 1
print("Pretraining finished, starting to train on specific data!")
step = 1
# Keep training until reach max iterations
batch_x, batch_y = ptr_x, ptrain_d
while step * 5000 < training_iters:
    # Reshape data to get 28 seq of 28 elements
    #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    # Run optimization op (backprop)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob:0.5})
    if step % display_step == 0:
        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
        # Calculate batch loss
        # loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        test_accuracy = sess.run(accuracy, feed_dict={x: tst_x, y: test_d, keep_prob:1.0})
        print("Iter " + str(step*120) + ", Minibatch Accuracy= " + \
              "{:.6f}".format(acc) +",Test Accuracy = " +  "{:.6f}".format(test_accuracy))
    step += 1
print("results look ok?")
