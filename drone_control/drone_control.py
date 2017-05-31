# This is the drone control functions to insert to the assembly
import logging
import time
from threading import Timer
import numpy as np
import cv2
from threading import Thread

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

    def Go_to(self, target, Commander, Kp=30, Ki=0, Kd=-1500):
        """ PID controller to get to specific location """
        if self.not_found_counter > 10:
            # drone lost
            Commander.send_setpoint(0, 0, 0, 0)
            print("shutting down")
        else:
            command = (target - self.loc) * Kp + self.vel * Kd
            pitch = min(max(command[0], -10), 10)
            roll = - min(max(command[1], -10), 10)
            thrust = min(max(command[2]*3000, -15000), 5000)+35000
            thrust = int(thrust)
            print(self.loc, self.vel, roll, pitch, thrust)
            self.cmd.send_setpoint(roll, pitch, 0, thrust)
        return

    def Start_up(self, thrust):
        self.cmd.send_setpoint(0, 0, 0, thrust)

    def get_loc(self, coordinates, read_failed, loc_prev, vel_prev):
        """
        Return a weighted sum of location from the camera
        and the previous momentum.
        """
        cam_coord = np.array(coordinates)
        if loc_prev.all() == None:
            return cam_coord, np.array([0, 0, 0])
        if read_failed[0] == 1:
            self.not_found_counter += 1
            return loc_prev+vel_prev, vel_prev
        self.not_found_counter = 0
        loc_current = 0.7*cam_coord+0.3*(loc_prev+vel_prev)
        return loc_current, (loc_current-loc_prev)

def simplified_control(targets, link_uri, start_signal):
    """
    The main control function, to be called as a separate thread from the gesture
    recognition part. This function continuously reads the targets and attemps to
    move the drone to the target.
    """
    if type(link_uri) == str:
        coordinates = [0, 0, 0]
        read_failed = [1]
        # Initialise
        camera_Thread = Thread(target=cam.simplified_loop,
                               args=(coordinates, read_failed))
        camera_Thread.start()
        cf = Drone(link_uri)
        cmd = cf.Initialise()
        input("press enter when ready")
        target_thread = Thread(target=change_location, args=(targets,))
        target_thread.start()
        input("press enter again")
        cf.Start_up(37500)
        while True:
            time.sleep(0.01)
            if read_failed[0] == 0:
                print("drone found")
                cmd.send_setpoint(0, 0, 0, 0)
                break
        while True:
            # updates the coordinate list from the camera feed
            # updates the drone location and velocity
            cf.loc, cf.vel = cf.get_loc(
                coordinates, read_failed, loc_prev=cf.loc, vel_prev=cf.vel)
            cf.Go_to(np.array(targets), cmd)
            time.sleep(0.01)
    if type(link_uri) == list:
        assert len(targets) == len(
            link_uri), "Provide exactly one link_uri for each target location"

def change_location(target):
    while True:
        target[0] = -target[0]
        print("AAAAAA")
        print("AAAAAA")
        print("AAAAAA")
        print("AAAAAA")
        print("AAAAAA")
        time.sleep(5)

def control(target, link_uri, start_signal):
    """
    The main control function, to be called as a separate thread from the gesture
    recognition part. This function continuously reads the targets and attemps to
    move the drone to the target.
    """
    if type(link_uri) == str:
        coordinates = [0, 0, 0]
        read_failed = [1]
        # Initialise
        camera_Thread = Thread(target=cam.simplified_loop,
                               args=(coordinates, read_failed))
        camera_Thread.start()
        cf = Drone(link_uri)
        cmd = cf.Initialise()
        while start_signal[0]==0:
            time.sleep(0.1)
        cf.Start_up(37500)

        target_thread = Thread(target=change_location, args = (target))
        target_thread.start()

        while True:
            time.sleep(0.01)
            if read_failed[0] == 0:
                print("drone found")
                cmd.send_setpoint(0, 0, 0, 0)
                break
        while True:
            # updates the coordinate list from the camera feed
            # updates the drone location and velocity
            cf.loc, cf.vel = cf.get_loc(
                coordinates, read_failed, loc_prev=cf.loc, vel_prev=cf.vel)
            cf.Go_to(np.array(target), cmd)
            time.sleep(0.01)
    if type(link_uri) == list:
        assert len(target) == len(
            link_uri), "Provide exactly one link_uri for each target location"

if __name__ == '__main__':
    target=[0.1,0,0]
    simplified_control(target, "radio://0/80/250K",[0])
