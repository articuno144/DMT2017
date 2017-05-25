# This is the drone control functions to insert to the assembly
import logging
import time
from threading import Timer
import numpy as np

import camera_functions as cam
import cflib
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig


def get_loc(vc1, ff1, vc2, ff2, vc3, ff3, a, b, c, loc_prev, vel_prev):
    """ 
    Return a weighted sum of location from the camera
    and the previous momentum.
    """
    x1, y1 = cam.frame_loc(vc1, ff1, 0)
    x2, y2 = cam.frame_loc(vc2, ff2, 0)
    x3, y3 = cam.frame_loc(vc3, ff3, 0)
    t1, t2, t3 = cam.get_angle(x1, y1), cam.get_angle(
        x2, y2), cam.get_angle(x3, y3)
    cam_coord = cam.get_coordinates(t1, t2, t3, a, b, c)
    loc_current = 0.7*cam_coord+0.3*(loc_prev+vel_prev)
    return loc_current, (loc_current-loc_prev)


class Drone():
    """
    Here we are only using the cameras and the onboard
    control system.
    """

    def __init__(self, link_uri):
        """ Initialises with the specific uri """
        self.loc = [None, None, None]  # location
        self.vel = [None, None, None]  # velocity
        self.loc = np.array(self.loc)
        self.vel = np.array(self.vel)
        # velocity here just mean movement between frames, not actual
        # velocity
        # location and velocity are stored as np arrays.
        self._cf = Crazyflie()
        self.uri = link_uri
        self.target = np.array([0, 0, 0])

    def Initialise(self):
        """ 
        Connect with the drone by the link uri, get initial
        location and velocity.
        """
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self._cf.open_link(self.uri)
        # move the line above to main function for multi drones
        self.cmd = cflib.crazyflie.Commander(self._cf)
        self.cmd.send_setpoint(0, 0, 0, 0)
        return self.cmd

    def Go_to(self, target, Kp, Ki, Kd):
        """ PID controller to get to specific location """
        #################################
        return

    def Update(self):
        """ Updates the drone location and velocity """
        loc_prev, vel_prev = self.loc, self.vel
        self.loc = get_loc(loc_prev, vel_prev)
        self.vel = (self.loc - loc_prev)
