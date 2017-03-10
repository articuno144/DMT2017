import logging
import time
from threading import Thread
import struct

import cflib
from cflib.crazyflie import Crazyflie

logging.basicConfig(level=logging.ERROR)

def Init(uri = "radio://0/80/250K"):
    cflib.crtp.init_drivers(enable_debug_driver=False)
    cf = Crazyflie()
    cf.open_link("radio://0/80/250K")
    cf.commander.send_setpoint(0,0,0,0)
    return cf
