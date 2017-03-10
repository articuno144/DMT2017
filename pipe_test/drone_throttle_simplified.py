import logging
import time
from threading import Thread
import struct

import cflib
from cflib.crazyflie import Crazyflie
from drone_simple import init_cf,send_setpoint

logging.basicConfig(level=logging.ERROR)

f = open(r'\\.\pipe\GesturePipe', 'r+b', 0)
x = time.time()
n = struct.unpack('I', f.read(4))[0]    # Read str length
s = f.read(n)                           # Read str
f.seek(0)                               # Important!!!
assert list(s)[0]>0, "Pipe reads 0 from MMG, please check the C# end"
print("connection successful, call pipe() to continue")
    

def pipe(s,f):
    cf = init_cf()
    for i in range(1200):
        s = bytes(s)
        f.write(struct.pack('I', len(s)) + s)   # Write str length and str
        f.seek(0)                               # EDIT: This is also necessary

        n = struct.unpack('I', f.read(4))[0]    # Read str length
        s = list(f.read(n))                           # Read str
        f.seek(0)                               # Important!!!
        roll = (s[0]-100)*10/3.14
        pitch = (s[1]-100)*10/3.14
        if i>1000:
            send_setpoint(cf,int(roll/2),int(pitch/2),thrust=35000)
        print("{:.4f}".format(roll), '   ', "{:.4f}".format(pitch))
    cf.close_link()
    
i = input("key in anything to fly")
while len(i)>1:
    pipe(s,f)
    i = input("key in anything to fly")
