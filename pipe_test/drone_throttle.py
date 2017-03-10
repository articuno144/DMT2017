import logging
import time
from threading import Thread
import struct

import cflib
from cflib.crazyflie import Crazyflie

logging.basicConfig(level=logging.ERROR)

global s

class MotorRampExample:
    """Example that connects to a Crazyflie and ramps the motors up/down and
    the disconnects"""

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """

        self._cf = Crazyflie()

        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)
        self.thrust = 0

        self._cf.open_link(link_uri)

        print('Connecting to %s' % link_uri)

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""

        # Start a separate thread to do the motor test.
        # Do not hijack the calling thread!
        Thread(target=self._thrust_motors).start()

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)
        
    def _thrust_motors(self):
        pitch = 0
        roll = 0
        yawrate = 0

        # Unlock startup thrust protection
        self._cf.commander.send_setpoint(0, 0, 0, 0)

        while True:
            self._cf.commander.send_setpoint(roll, pitch, yawrate, self.thrust)
            time.sleep(0.1)

    def _ramp_motors(self):
        thrust_mult = 1
        thrust_step = 500
        thrust = 20000
        pitch = 0
        roll = 0
        yawrate = 0

        # Unlock startup thrust protection
        self._cf.commander.send_setpoint(0, 0, 0, 0)

        while thrust >= 20000:
            self._cf.commander.send_setpoint(roll, pitch, yawrate, thrust)
            time.sleep(0.1)
            if thrust >= 25000:
                thrust_mult = -1
            thrust += thrust_step * thrust_mult
        self._cf.commander.send_setpoint(0, 0, 0, 0)
        # Make sure that the last packet leaves before the link is closed
        # since the message queue is not flushed before closing
        time.sleep(0.1)
        self._cf.close_link()

def initialise()
    f = open(r'\\.\pipe\GesturePipe', 'r+b', 0)
    x = time.time()
    n = struct.unpack('I', f.read(4))[0]    # Read str length
    s = f.read(n)                           # Read str
    f.seek(0)                               # Important!!!
    assert list(s)[0]>0, "Pipe reads 0 from MMG, please check the C# end"
    cflib.crtp.init_drivers(enable_debug_driver=False)
    
def pipe():
    mre = MotorRampExample("radio://0/80/250K")
    for i in range(1000):
        s = list(s)
        s = bytes(s)
        f.write(struct.pack('I', len(s)) + s)   # Write str length and str
        f.seek(0)                               # EDIT: This is also necessary

        n = struct.unpack('I', f.read(4))[0]    # Read str length
        s = f.read(n)                           # Read str
        f.seek(0)                               # Important!!!
        mre.thrust = list(s)[0]*100
        print(list(s)[0], '   ', list(s)[0]*100)
