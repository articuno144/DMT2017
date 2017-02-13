# adapted from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import math
import numpy as np

print('program starting ...')

desired_dataframe = pd.read_csv('d2.csv',sep = ',', header = None)
desired_values = desired_dataframe.values[:,0]
desired_values = np.transpose(desired_values.reshape((6,-1)))

input_dataframe = pd.read_csv('x2.csv',sep = ',',header = None)
input_values = np.array(input_dataframe.values[:,0],dtype = 'float')
input_values = input_values.reshape((5,300,-1))

noise_dataframe = pd.read_csv('noise.csv',sep= ',', header= None)
noise_values = np.array(noise_dataframe.values[:,0],dtype = 'float')
noise_values = noise_values.reshape((5,300,-1))

noised_dataframe = pd.read_csv('noised.csv',sep= ',', header= None)
noised_values = noised_dataframe.values[:,0]
noised_values = np.transpose(noised_values.reshape((6,-1)))

noiset_dataframe = pd.read_csv('noiset.csv',sep= ',', header= None)
noiset_values = np.array(noiset_dataframe.values[:,0],dtype = 'float')
noiset_values = noiset_values.reshape((5,300,-1))

noisetd_dataframe = pd.read_csv('noisetd.csv',sep= ',', header= None)
noisetd_values = noisetd_dataframe.values[:,0]
noisetd_values = np.transpose(noisetd_values.reshape((6,-1)))

specif_train_dataframe = pd.read_csv('input_myt0130.csv',sep = ',',header = None)
specif_train_values = np.array(specif_train_dataframe.values[:,0],dtype = 'float')
specif_train_values = specif_train_values.reshape((5,300,-1))

test_dataframe = pd.read_csv('input_myt0130t.csv',sep = ',',header = None)
test_values = np.array(test_dataframe.values[:,0],dtype = 'float')
test_values = test_values.reshape((5,300,-1))

testd_dataframe = pd.read_csv('desired_myt0130.csv',sep= ',', header= None)
test_d = testd_dataframe.values[:,0]
test_d = np.transpose(test_d.reshape((6,-1)))
ptrain_d = test_d

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
for j in range(37000):
    noise_values[:,:,j] = standardize_set(noise_values[:,:,j])
for j in range(5500):
    noiset_values[:,:,j] = standardize_set(noiset_values[:,:,j])
for j in range(320):
    specif_train_values[:,:,j] = standardize_set(specif_train_values[:,:,j])
for j in range(320):
    test_values[:,:,j] = standardize_set(test_values[:,:,j])
batch_size = 1000
#train_idx = 3071
#ptrain_idx = 3191
#test_size = total_size - ptrain_idx
#train_x = input_values[:,:,:train_idx]
train_x = np.concatenate((input_values,noise_values),2)
#train_d = desired_values[:train_idx,:]
train_d = np.concatenate((desired_values,noised_values),0)
#ptrain_x = input_values[:,:,train_idx:ptrain_idx]
#ptrain_d = desired_values[train_idx:ptrain_idx,:]
#test_x = input_values[:,:,ptrain_idx:]
#test_d = desired_values[ptrain_idx:,:]



##tr_x = np.empty([300,5,batch_size])
##for i in range(batch_size):
##    tr_x[:,:,i] = np.transpose(train_x[:,:,i],[1,0,2])
##ptr_x = np.empty([300,5,120])
##for i in range(120):
##    ptr_x[:,:,i] = np.transpose(ptrain_x[:,:,i],[1,0,2])
##tst_x = np.empty([300,5,test_size])
##for i in range(test_size):
##    tst_x[:,:,i] = np.transpose(test_x[0,:,i],[1,0,2])
tr_x = np.transpose(train_x,[2,1,0])
ptr_x = np.transpose(specif_train_values,[2,1,0])
tst_x = np.transpose(test_values,[2,1,0])
tstn_x = np.transpose(noiset_values,[2,1,0])
noise_values = np.transpose(noise_values,[2,1,0])
#now the input are of size [batch_size, n_input]

learning_rate = tf.placeholder("float")
training_iters = 2000000
display_step = 5

# Network Parameters
n_input = 300 # input size, currently only using col 3,4 (4,5 in matlab)
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_classes = 6 # 5 gestures, 1 noise

# tf Graph input
x = tf.placeholder("float", [None, n_input,5])
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
    x = tf.reshape(x, shape=[-1, n_input,5, 1])
    x1 = x[:,:,:3,:]
    x2 = x[:,:,3:,:]
    # Convolution Layer
    conv1 = conv2d(x1, weights['wc1'], biases['bc1'])
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

    # Convolution Layer
    conv3 = conv2d(x2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)
    # Convolution Layer
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=2)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc2 = tf.reshape(conv4, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc = tf.concat(1,[fc1,fc2])
    # Apply Dropout
    fc = tf.nn.dropout(fc, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc, weights['out']), biases['out'])
    return out

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 3, 1, 64])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 1, 64, 128])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([73*128, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'wc3': tf.Variable(tf.random_normal([3, 2, 1, 64])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc4': tf.Variable(tf.random_normal([5, 1, 64, 128])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd2': tf.Variable(tf.random_normal([73*128, 1024])),
    'out': tf.Variable(tf.random_normal([2048 , n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bc4': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
second_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=[weights['wd1'],weights['wd2'],weights['out'],biases['bd1'],biases['bd2'],biases['out']])
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
p = 0
# Keep training until reach max iterations
while step * batch_size < training_iters:
    batch_x, batch_y = tr_x[p:p+batch_size,:,:], train_d[p:p+batch_size,:]
    p+=batch_size
    if p>=40000:
        p=0
    # Reshape data to get 28 seq of 28 elements
    #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    # Run optimization op (backprop)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob:0.5,learning_rate: 0.005})
    if step % display_step == 0:
        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
        # Calculate batch loss
        # loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        test_accuracy = sess.run(accuracy, feed_dict={x:tst_x,y:test_d,keep_prob:1.0})
        noise_accuracy = sess.run(accuracy, feed_dict={x: tstn_x, y: noisetd_values, keep_prob:1.0})
        print("Iter " + str(step*120) + ", Minibatch Accuracy= " + \
              "{:.6f}".format(acc) +",Test Accuracy = " +  "{:.6f}".format(test_accuracy)+" ,Noise Accuracy = " + "{:.6f}".format(noise_accuracy))
    step += 1
print("Pretraining finished, starting to train on specific data!")
step = 1

n = 0
# Keep training until reach max iterations
batch_x, batch_y = ptr_x, ptrain_d
while step * 1000 < training_iters:
    noise_x, noise_y = noise_values[n:n+batch_size,:,:], noised_values[n:n+batch_size,:]
    n+=batch_size
    if n>=30000:
        n=0
    # Reshape data to get 28 seq of 28 elements
    #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    # Run optimization op (backprop)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob:0.5,learning_rate : 0.0001})
    sess.run(optimizer, feed_dict={x: noise_x, y: noise_y, keep_prob:0.5,learning_rate : 0.0002})
    if step % display_step == 0:
        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
        # Calculate batch loss
        # loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        test_accuracy = sess.run(accuracy, feed_dict={x:tst_x,y:test_d,keep_prob:1.0})
        noise_accuracy = sess.run(accuracy, feed_dict={x: tstn_x, y: noisetd_values, keep_prob:1.0})
        print("Iter " + str(step*120) + ", Minibatch Accuracy= " + \
              "{:.6f}".format(acc) +",Test Accuracy = " +  "{:.6f}".format(test_accuracy)+" ,Noise Accuracy = " + "{:.6f}".format(noise_accuracy))
    step += 1
noise_preds = sess.run(tf.argmax(pred,1),feed_dict = {x:tstn_x, keep_prob:1.0})
noise_d = sess.run(tf.argmax(noisetd_values,1))
gest_preds = sess.run(tf.argmax(pred,1),feed_dict = {x:tst_x, y :test_d,keep_prob:1.0})
gest_d = sess.run(tf.argmax(test_d,1))


def tell_diff(arr1, arr2):
    if len(arr1)==len(arr2):
        m = {}
        for i in range(len(arr1)):
            if arr1[i]!=arr2[i]:
                if (arr1[i],arr2[i]) in m.keys():
                    m[(arr1[i],arr2[i])]+=1
                else:
                    m[(arr1[i],arr2[i])]=1 
                print(arr1[i], " => ",arr2[i], i)
        print(m)

tell_diff(gest_d,gest_preds)
tell_diff(noise_d,noise_preds)
print("results look ok?")
