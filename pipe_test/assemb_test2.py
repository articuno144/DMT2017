import time
import struct
import tensorflow as tf
import numpy as np
import random
import time


f = open(r'\\.\pipe\GesturePipe', 'r+b', 0)
x = time.time()
n = struct.unpack('I', f.read(4))[0]    # Read str length
s = f.read(n)                           # Read str
f.seek(0)                               # Important!!!
print('Read:', s)
m = np.zeros([6,300])
ctr = 0

def process(m):
    "m is a 6 by 300 matrix"
    return [42,random.randint(1,5)]            #pretending to process data here

x = time.time()

while True:
    s = list(s)
    m[:,:299] = m[:,1:]
    m[:,299] = s
    p = process(m)
    p = bytes(p)
    f.write(struct.pack('I', len(p)) + p)   # Write str length and str
    f.seek(0)                               # EDIT: This is also necessary

    n = struct.unpack('I', f.read(4))[0]    # Read str length
    s = f.read(n)                           # Read str
    f.seek(0)                               # Important!!!
    s = list(s)
    ctr+=1
    if ctr==100:
        ctr=0
        print(time.time()-x)
        x = time.time()
