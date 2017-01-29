import time
import struct

f = open(r'\\.\pipe\NPtest', 'r+b', 0)
s = b'Message[0]'

while True:
    i = int(s[8:-1])+1
    s = s[0:8]+'{0}'.format(i).encode('ascii')+s[-1:] 
    f.write(struct.pack('I', len(s)) + s)   # Write str length and str
    f.seek(0)                               # EDIT: This is also necessary
    print('Wrote:', s)

    n = struct.unpack('I', f.read(4))[0]    # Read str length
    s = f.read(n)                           # Read str
    f.seek(0)                               # Important!!!
    print('Read:', s)

##    time.sleep(2)

##while True:
##    s = 
##
##    if i ==0:
##        i+=1
##        f.write(struct.pack('IIII',len(init),init[0],init[1],init[2]))
##    else:
##        f.write(struct.pack('IIII',len(s),s[0],s[1],s[2]))   # Write str length and str
##    f.seek(0)                               # EDIT: This is also necessary
##    print('Wrote:', s)
##    
##
##    n = struct.unpack('I', f.read(4))[0]    # Read str length
##    s = f.read(n)                           # Read str
##    f.seek(0)                               # Important!!!
##    print('Read:', s)
##
##    time.sleep(2)
