import time
import struct
import tensorflow as tf
import numpy as np
import random
import time
from CNN_functions import conv_net


x = tf.placeholder("float",[1,300,6])
keep_prob = tf.placeholder("float")

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 3, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 1, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([73*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'wc3': tf.Variable(tf.random_normal([3, 3, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc4': tf.Variable(tf.random_normal([5, 1, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd2': tf.Variable(tf.random_normal([73*64, 1024])),
    'out': tf.Variable(tf.random_normal([2048 , 9]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([32])),
    'bc4': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([9]))
}

pred = conv_net(x, weights, biases, keep_prob)
with tf.device('/cpu:0'):
	c_argmax = tf.argmax(pred,1)
	c_sigmoid_pred = tf.sigmoid(pred)

m = np.zeros([1,300,6],dtype = float)
ctr = 0

def process(sess,m):
    "m is a 6 by 300 matrix"
    return list([0,int(sess.run(c_argmax,feed_dict = {x:m,keep_prob:1.0}))])

def standardize(s):
    s = list(s)
    assert len(s)==12 , "LengthError"
    ss = []
    ss.append((s[0]*256+s[1]-125)/100)
    ss.append((s[2]*256+s[3]-125)/100)
    ss.append((s[4]*256+s[5]-125)/100)
    ss.append((s[6]*256+s[7]-250)/100)
    ss.append((s[8]*256+s[9]-250)/100)
    ss.append((s[10]*256+s[11]-250)/100)
    return ss


sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess,'Saved\\CNN_pretrain')

f = open(r'\\.\pipe\GesturePipe', 'r+b', 0)
n = struct.unpack('I', f.read(4))[0]    # Read str length
s = f.read(n)                           # Read str
f.seek(0)                               # Important!!!
print('Read:', list(s))



##curTime = time.time()
while True:
    s = list(s)
    m[0,:299,:] = m[0,1:,:]
    m[0,299,:] = s
    p = process(sess,m)
    p = bytes(p)
    f.write(struct.pack('I', len(p)) + p)   # Write str length and str
    f.seek(0)                               # EDIT: This is also necessary

    n = struct.unpack('I', f.read(4))[0]    # Read str length
    s = f.read(n)                           # Read str
    f.seek(0)                               # Important!!!
    s = standardize(s)
    print('Read:',s)
    print('pred: ',p[1])
##    ctr+=1
##    if ctr==100:
##        ctr=0
##        print(time.time()-curTime)
##        curTime = time.time()
