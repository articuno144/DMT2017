import logging
import time
from threading import Thread
import struct
from math import sin, cos

import cflib
from cflib.crazyflie import Crazyflie

logging.basicConfig(level=logging.ERROR)

def init_cf(uri="radio://0/80/250K"):
    cflib.crtp.init_drivers(enable_debug_driver=False)
    cf = Crazyflie()
    cf.open_link("radio://0/80/250K")
    cf.commander.send_setpoint(0,0,0,0)
    return cf

def circle(cf):
    for i in range(36):
        pitch = 10*sin(i*10*3.14/180)
        roll = 10*cos(i*10*3.14/180)
        cf.commander.send_setpoint(roll,pitch,0,32000)
        time.sleep(0.1)

def send_setpoint(cf,roll,pitch,yaw=0,thrust=32000):
    cf.commander.send_setpoint(roll,pitch,yaw,thrust)
