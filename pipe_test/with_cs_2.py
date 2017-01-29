import time
import struct

f = open(r'\\.\pipe\NPtest', 'r+b', 0)

n = struct.unpack('I', f.read(4))[0]    # Read str length
s = f.read(n)                           # Read str
f.seek(0)                               # Important!!!
print('Read:', s)

while True:
    s = list(s)
    for i in range(len(s)):
        s[i]+=i
    s = bytes(s)
    f.write(struct.pack('I', len(s)) + s)   # Write str length and str
    f.seek(0)                               # EDIT: This is also necessary
    print("Wrote:",list(s))

    n = struct.unpack('I', f.read(4))[0]    # Read str length
    s = f.read(n)                           # Read str
    f.seek(0)                               # Important!!!
    print('Read:', list(s))

    time.sleep(2)
