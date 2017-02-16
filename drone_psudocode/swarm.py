import math
from math import pi

class drone:
    def __init__(self,tag,loc):
        self.tag = tag
        self.loc = loc
        self.angle = [0,0]
        self.omega = [0,0]
        
    def apply_torque(self,domega):
        self.omega[0]+=domega[0]
        self.omega[1]+=domega[1]

    def change_angle(self):
        self.angle[0]+=self.omega[0]
        self.angle[1]+=self.omega[1]

class swarm:
    centre = [0,0]
    def __init__(self,num_drones):
        self.n_drones = num_drones
        self.target = []
        for i in range(num_drones):
            self.target.append([math.sin(2*pi*i/num_drones),math.cos(2*pi*i/num_drones)])
        self.loc = self.target
        self.
