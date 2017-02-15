import time
import struct
import tensorflow as tf
import numpy as np
import random
import time
from CNN_functions_MMGonly import conv_net


x = tf.placeholder("float",[1,50,3])
keep_prob = tf.placeholder("float")

weights = {
    'wc3': tf.Variable(tf.random_normal([3, 3, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc4': tf.Variable(tf.random_normal([5, 1, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd2': tf.Variable(tf.random_normal([10*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024 , 9]))
}

biases = {
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

m = np.zeros([1,50,3],dtype = float)
ctr = 0

def process(sess,m):
    "m is a 6 by 50 matrix"
    return list([0,int(sess.run(c_argmax,feed_dict = {x:m,keep_prob:1.0}))])

def standardize(s):
    s = list(s)
    assert len(s)==6 , "LengthError"
    ss = []
    ss.append((s[0]*256+s[1]-250)/100)
    ss.append((s[2]*256+s[3]-250)/100)
    ss.append((s[4]*256+s[5]-250)/100)
    return ss


sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess,'Saved\\CNN_MMGonly50_pretrain')

f = open(r'\\.\pipe\GesturePipe', 'r+b', 0)
n = struct.unpack('I', f.read(4))[0]    # Read str length
s = f.read(n)                           # Read str
f.seek(0)                               # Important!!!
print('Read:', list(s))



##curTime = time.time()
while True:
##    ctr+=1
    s = list(s)
    m[0,:49,:] = m[0,1:,:]
    m[0,49,:] = standardize(s)
    p = process(sess,m)
    p = bytes(p)
    f.write(struct.pack('I', len(p)) + p)   # Write str length and str
    f.seek(0)                               # EDIT: This is also necessary

    n = struct.unpack('I', f.read(4))[0]    # Read str length
    s = f.read(6)                           # Read str
    f.seek(0)                               # Important!!!
##    print(ctr,'\t Read:',list(s))
    print('pred: ',p[1])
##    ctr+=1
##    if ctr==100:
##        ctr=0
##        print(time.time()-curTime)
##        curTime = time.time()
