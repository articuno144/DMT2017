import time
import struct
import tensorflow as tf
import numpy as np
import random
import time
from threading import Thread

import drone_control as dc
from CNN_functions_MMGonly import conv_net

n_classes = 9  # 5 gestures, 1 noise
x = tf.placeholder("float", [None, 50, 3])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder("float")
n_gesture = 3
n_trial = 5
smp_per_trial = 8
ready = 0
new_noise = None
training_set = None

roll, pitch, yaw = None, None, None

weights = {
    'wc3': tf.Variable(tf.random_normal([3, 3, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc4': tf.Variable(tf.random_normal([5, 1, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd2': tf.Variable(tf.random_normal([10*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 9]))
}

biases = {
    'bc3': tf.Variable(tf.random_normal([32])),
    'bc4': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([9]))
}

pred = conv_net(x, weights, biases, keep_prob)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
with tf.device('/cpu:0'):
    c_argmax = tf.argmax(pred, 1)
    c_sigmoid_pred = tf.sigmoid(pred)

m = np.zeros([1, 50, 3], dtype=float)
ctr = 0


def process(sess, m):
    "m is a 6 by 50 matrix"
    return list([ready, int(sess.run(c_argmax, feed_dict={x: m, keep_prob: 1.0}))])


def standardize(s):
    s = list(s)
    assert len(s) == 6, "LengthError"
    ss = []
    ss.append((s[0]*256+s[1]-250)/100)
    ss.append((s[2]*256+s[3]-250)/100)
    ss.append((s[4]*256+s[5]-250)/100)
    return ss

learning_rate = tf.placeholder("float")
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, 'Saved\\CNN_MMGonly50_yuntao1')

f = open(r'\\.\pipe\GesturePipe', 'r+b', 0)
n = struct.unpack('I', f.read(4))[0]    # Read str length
s = f.read(n)                           # Read str
f.seek(0)
print('Read:', list(s))


target = [[0.15, 0, -0.05], [-0.15, 0, -0.15]]
start_signal = [0]
target_locked = [True, True]
control_Thread = Thread(target=dc.control,
                        args=(target, ["radio://0/80/250K", "radio://0/12/1M"], start_signal))
control_Thread.start()
new_gesture_counter = 0

status = 'low'

def enter_start(start_signal):
    while True:
        time.sleep(0.1)
        input("press enter to start or stop")
        start_signal[0] = 1 - start_signal[0]

enter_start_thread = Thread(target=enter_start, args=(start_signal,))
enter_start_thread.start()

gesture_window = [8, 8, 8, 8, 8, 8, 8, 8]
while True:
   # print(start_signal, target_locked, target)
    s = list(s)
    roll = (s[9]-100)*10/3.14
    pitch = (s[10]-100)*10/3.14
    m[0, :49, :] = m[0, 1:, :]
    m[0, 49, :] = standardize(s[3:9])
    p = process(sess, m)
    p = bytes(p)
    f.write(struct.pack('I', len(p)) + p)   # Write str length and str
    f.seek(0)

    n = struct.unpack('I', f.read(4))[0]    # Read str length
    s = f.read(n)                           # Read str
    f.seek(0)
    gesture_window[:7] = gesture_window[1:]
    gesture_window[7] = p[1]
# make sure commands are not executed twice
    if new_gesture_counter > 0:
        new_gesture_counter += 1
    if new_gesture_counter > 40:
        new_gesture_counter = 0
    if all(pred == 8 for pred in gesture_window[4:]) and gesture_window[3] != 8:
        # change drone commands based on the gesture, can be changed easily
        if new_gesture_counter == 0:
            new_gesture_counter += 1
        if gesture_window[0] == 0:
            print("00000")
            print("00000")
            print("00000")
            print("00000")            
            if all(locked == False for locked in target_locked):
                target_locked[:] = [True, True]
        elif gesture_window[0] == 1:
            print("11111")
            print("11111")
            print("11111")
            print("11111")  
            if target_locked[0]==target_locked[1]:
                target_locked[1]=not target_locked[1]
            for i in range(len(target_locked)):
                target_locked[i] = not target_locked[i]
        elif gesture_window[0] == 2:
            print("22222")
            print("22222")
            print("22222")
            print("22222")
            for locked in target:
                locked[2] = 0 - locked[2]
            if status=='low':
                status = 'high'
            else:
                status = 'low'
    if not target_locked[0]:
        target[0][0] = min(max(target[0][0]+pitch/1000, -0.2), 0.2)
        target[0][1] = min(max(target[0][1]+roll/1000, -0.2), 0.2)
    if not target_locked[1]:
        target[1][0] = min(max(target[1][0]+pitch/1000, -0.2), 0.2)
        target[1][1] = min(max(target[1][1]+roll/1000, -0.2), 0.2)
    buf = p[1]
    print(buf,target_locked, status, target[0], target[1], roll, pitch)
