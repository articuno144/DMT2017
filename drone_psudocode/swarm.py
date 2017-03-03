import math
from math import pi

class drone:
    def __init__(self,tag = None,loc = [3.,3.]):
        self.tag = tag
        self.loc = loc
        self.angle = [0.,0.]
        self.omega = [0.,0.]
        
    def apply_torque(self,domega):
        self.omega[0]+=domega[0]
        self.omega[1]+=domega[1]

    def change_angle(self):
        self.angle[0]+=self.omega[0]
        self.angle[1]+=self.omega[1]

    def follow_target(self,target):
        #the drone follows the target, dynamics goes here
        pass

class swarm:
    centre = [0.,0.]
    def __init__(self,num_drones):
        self.n_drones = num_drones
        self.target = []
        self.drones = []
        self.formation = 0
        for i in range(num_drones):
            loc = [math.sin(2*pi*i/num_drones),math.cos(2*pi*i/num_drones)]
            self.target.append(loc)
            self.drones.append(drone(i,loc))
    def __str__(self):
        return str(self.n_drones)+" drones centred at "+str(self.centre[0])+","+str(self.centre[1])+" with formation "+str(self.formation)
        
    def add_drone(self,drn = drone(0,[3.,3.])):
        drn.tag = self.n_drones
        self.drones.append(drn)
        self.n_drones+=1
        self.target = self.update_target()
    
    def update_target(self):
        assert self.n_drones>1, "too few drones"
        self.target = []
        if self.formation == 0:
            for i in range(self.n_drones):
                loc = [math.sin(2*pi*i/self.n_drones)+self.centre[0],math.cos(2*pi*i/self.n_drones)+self.centre[1]]
                self.target.append(loc)
        if self.formation == 1:
            for i in range(self.n_drones):
                loc = [self.centre[0],self.centre[1]+2-4.*i/(self.n_drones-1)]
                self.target.append(loc)
        self.allocate_targets()
        return self.target
        
    def allocate_targets(self):
        #matches the target order to drone order TO DO, the swarm intellegence thing, maybe change drone order instead?
        pass
    
    def move_loc(self,target,move):
        target[0]+=move[0]
        target[1]+=move[1]

    def move(self,move):
        self.centre[0]+=move[0]
        self.centre[1]+=move[1]
        for target in self.target:
            self.move_loc(target,move)
            
    def get_drone(self,tag):
        return self.drones[tag]

    def change_formation(self,new_formation):
        self.formation = new_formation
        self.update_target()
    
    def update(self,time_step):
        #update drone locations and centre
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    