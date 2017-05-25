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
        self.loc_prev = None
        self.vel_prev = None

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

    def Go_to(self, target, Commander, Kp=1, Ki=0, Kd=0):
        """ PID controller to get to specific location """
        #################################
        return

    def get_loc(coordinates, read_failed, loc_prev=None, vel_prev=None):
        """ 
        Return a weighted sum of location from the camera
        and the previous momentum.
        """
        cam_coord = np.array(coordinates)
        if loc_prev == None:
            return cam_coord, np.array([0, 0, 0])
        if read_failed[1]:
            return loc_prev+vel_prev
        loc_current = 0.7*cam_coord+0.3*(loc_prev+vel_prev)
        return loc_current, (loc_current-loc_prev)


def control(target, link_uri):
    """
    The main control function, to be called as a separate thread from the gesture
    recognition part. This function continuously reads the targets and attemps to
    move the drone to the target.
    """
    if type(link_uri) == str:
        coordinates = [0, 0, 0]
        read_failed = [0]
        # Initialise
        vc0, vc1, vc2, first_frame0, first_frame1, first_frame2 = cam.Init()
        cf = Drone("radio://0/80/250K")
        cmd = cf.Initialise()
        while True:
            # updates the coordinate list from the camera feed
            cam.Cam(coordinates, read_failed, vc0, vc1, vc2,
                    first_frame0, first_frame1, first_frame2)
            # updates the drone location and velocity
            cf.loc, cf.vel = cf.get_loc(coordinates, read_failed, cf.loc, cf.vel)
            cf.Go_to(target, cmd)
    if type(link_uri) == list:
        assert len(target) == len(
            link_uri), "Provide exactly one link_uri for each target location"