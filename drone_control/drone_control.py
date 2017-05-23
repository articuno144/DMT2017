# This is the drone control functions to insert to the assembly
import logging
import time
from threading import Timer
import numpy as np

# import camera functions
#################################
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig


def get_loc(loc_prev, vel_prev):
    """ 
    Return a weighted sum of location from the camera
    and the previous momentum.
    """
    #################################
    return


class Drone():
    """
    Here we are only using the cameras and the onboard
    control system.
    """

    def __init__(self, link_uri):
        """ Initialises with the specific uri """
        self.dt = 2**-7
        self.loc = [None, None, None]  # location
        self.vel = [None, None, None]  # velocity
        self.loc = np.array(self.loc)
        self.vel = np.array(self.vel)
        # location and velocity are stored as np arrays.
        self._cf = Crazyflie()
        self._cf.open_link(link_uri)
        self.target = np.array([0, 0, 0])

    def Initialise(self):
        """ 
        Connect with the drone by the link uri, get initial
        location and velocity.
        """
        #################################
        return

    def Go_to(self, target, Kp, Ki, Kd):
        """ PID controller to get to specific location """
        #################################
        return

    def Update(self):
        """ Updates the drone location and velocity """
        loc_prev, vel_prev = self.loc, self.vel
        self.loc = get_loc(loc_prev, vel_prev)
        self.vel = (self.loc - loc_prev) / self.dt
