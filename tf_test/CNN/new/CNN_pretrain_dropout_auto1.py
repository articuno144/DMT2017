# adapted from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import math
import numpy as np
import k_mean

print('program starting ...')

desired_dataframe = pd.read_csv('d_pretrain.csv',sep = ',', header = None)
desired_values = desired_dataframe.values[:,0]
desired_values = np.transpose(desired_values.reshape((5,-1)))

input_dataframe = pd.read_csv('x_pretrain.csv',sep = ',',header = None)
input_values = np.array(input_dataframe.values[:,0],dtype = 'float')
input_values = input_values.reshape((5,300,-1))

t_desired_dataframe = pd.read_csv('test_desired.csv',sep = ',', header = None)
t_desired_values = t_desired_dataframe.values[:,0]
t_desired_values = np.transpose(t_desired_values.reshape((5,-1)))

t_input_dataframe = pd.read_csv('test_input.csv',sep = ',',header = None)
t_input_values = np.array(t_input_dataframe.values[:,0],dtype = 'float')
t_input_values = t_input_values.reshape((5,300,-1))
#standardize the person-specific inputs
def standardize_set(arr):
    a0 = np.multiply(np.add(arr[0,:],-125),0.01)
    a1 = np.multiply(np.add(arr[1,:],-125),0.01)
    a2 = np.multiply(np.add(arr[2,:],-125),0.01)
    a3 = np.multiply(np.add(arr[3,:],-250),0.01)
    a4 = np.multiply(np.add(arr[4,:],-250),0.01)
    return np.array([a0,a1,a2,a3,a4])

def generate_specific_sets(inputs,mean_indices,gesture,post_x,post_d):
    #append the right inputs to post_x and outputs to post_d, inputs takes the shape like in1, in2 ...
    for index in mean_indices:
        for i in range(8):
            _input = format_to_600(inputs,max(int(index)+i-4,300)).reshape([1,600])
            _desired = np.zeros(5).reshape([1,5])
            #gesture is currently 1-indexing
            _desired[0,(gesture-1)] = 1
            #add to post_x and post_d
            if post_x == None or post_d == None:
                return _input,_desired
            else:
                return np.concatenate((post_x,_input),axis = 0),np.concatenate((post_d,_desired),axis = 0)

#take sample from the current user
in1 = standardize_set(pd.read_csv('s9g1.csv',sep = ',',header = None).values.transpose())
in2 = standardize_set(pd.read_csv('s9g2.csv',sep = ',',header = None).values.transpose())
in3 = standardize_set(pd.read_csv('s9g3.csv',sep = ',',header = None).values.transpose())
in4 = standardize_set(pd.read_csv('s9g4.csv',sep = ',',header = None).values.transpose())
in5 = standardize_set(pd.read_csv('s9g5.csv',sep = ',',header = None).values.transpose())

len1 = in1.shape[1]
len2 = in2.shape[1]
len3 = in3.shape[1]
len4 = in4.shape[1]
len5 = in5.shape[1]

def format_to_600(x,i):
    return np.concatenate([x[3,i-300:i],x[4,i-300:i]])

five, threehundred, total_size = input_values.shape
for j in range(total_size):
    input_values[:,:,j] = standardize_set(input_values[:,:,j])
test_size = t_input_values.shape[2]
for j in range(test_size):
    t_input_values[:,:,j] = standardize_set(t_input_values[:,:,j])

train_idx = batch_size = 3071
train_x = input_values[:,:,:train_idx]
train_d = desired_values[:train_idx,:]
test_x = t_input_values
test_d = t_desired_values
cost = np.empty([700,5])
all_pred = np.empty([700,5])

tr_x = np.empty([600,batch_size])
for i in range(batch_size):
    tr_x[:,i] = np.concatenate([train_x[3,:,i],train_x[4,:,i]])
tr_x = np.transpose(tr_x)
#now the input are of size [batch_size, n_input]
t_x = np.empty([600,test_size])
for i in range(test_size):
    t_x[:,i] = np.concatenate([test_x[3,:,i],test_x[4,:,i]])
t_x = np.transpose(t_x)

learning_rate = 0.005
training_iters = 500000
display_step = 5

# Network Parameters
n_input = 600 # input size, currently only using col 3,4 (4,5 in matlab)
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_classes = 5 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder("float")
# Create model
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, n_input,1, 1])

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
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 1, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 1, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([150*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
pred_softmax = tf.nn.softmax(pred)
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

pred1 = []
pred2 = []
pred3 = []
pred4 = []
pred5 = []

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
            acc = sess.run(accuracy, feed_dict={x: t_x, y: test_d, keep_prob:1.0})
            # Calculate batch loss
            # loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Accuracy= " + \
                  "{:.6f}".format(acc))
        step += 1
    print("Pretraining finished, starting to train on specific data!")
#use pretraining data to find where the gestures are, as lists of indices
    for i in range(len1-300):
        t_i = i+300
        _input = format_to_600(in1,t_i).reshape([-1,600])
        _pred = sess.run(pred_softmax, feed_dict = {x: _input, keep_prob:1.0})
        if _pred.reshape([-1,5])[0][0]>=0.9:
            pred1.append(i+300)
    for i in range(len2-300):
        t_i = i+300
        _input = format_to_600(in2,t_i).reshape([-1,600])
        _pred = sess.run(pred_softmax, feed_dict = {x: _input, keep_prob:1.0})
        if _pred.reshape([-1,5])[0][1]>=0.9:
            pred2.append(i+300)
    for i in range(len3-300):
        t_i = i+300
        _input = format_to_600(in3,t_i).reshape([-1,600])
        _pred = sess.run(pred_softmax, feed_dict = {x: _input, keep_prob:1.0})
        if _pred.reshape([-1,5])[0][2]>=0.9:
            pred3.append(i+300)
    for i in range(len4-300):
        t_i = i+300
        _input = format_to_600(in4,t_i).reshape([-1,600])
        _pred = sess.run(pred_softmax, feed_dict = {x: _input, keep_prob:1.0})
        if _pred.reshape([-1,5])[0][3]>=0.9:
            pred4.append(i+300)
    for i in range(len5-300):
        t_i = i+300
        _input = format_to_600(in5,t_i).reshape([-1,600])
        _pred = sess.run(pred_softmax, feed_dict = {x: _input, keep_prob:1.0})
        if _pred.reshape([-1,5])[0][4]>=0.9:
            pred5.append(i+300)
    #find 3 mean locations for each index list
    means1 = np.floor(k_mean.k_mean_index(np.array(pred1),3))
    means2 = np.floor(k_mean.k_mean_index(np.array(pred2),4))
    means3 = np.floor(k_mean.k_mean_index(np.array(pred3),3))
    means4 = np.floor(k_mean.k_mean_index(np.array(pred4),3))
    means5 = np.floor(k_mean.k_mean_index(np.array(pred5),3))
    print(means1,means2,means3,means4,means5)
    means1 = [546,781]
    means2 = [315,522,754,986]
    means3 = [243,503,762]
    means4 = [741,1000,1250]
    means5 = [694,937,1164]
    #use the locations to generate training sets
    post_x = None
    post_d = None
    post_x,post_d = generate_specific_sets(in1,means1,1,post_x,post_d)
    post_x,post_d = generate_specific_sets(in2,means2,2,post_x,post_d)
    post_x,post_d = generate_specific_sets(in3,means3,3,post_x,post_d)
    post_x,post_d = generate_specific_sets(in4,means4,4,post_x,post_d)
    post_x,post_d = generate_specific_sets(in5,means5,5,post_x,post_d)
    #while loop to train the network with the specific set
    step = 1
    while step<1000:
        batch_x, batch_y = post_x, post_d
        # Reshape data to get 28 seq of 28 elements
        #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob:0.5})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: t_x, y: test_d, keep_prob:1.0})
            # Calculate batch loss
            # loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Accuracy= " + \
                  "{:.6f}".format(acc))
        step += 1
    print("All Done!")
