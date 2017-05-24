# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:06:36 2017

@author: david
"""
#rough work on coordinates

import numpy as np

#cam1 = yc1, zc1, FOV1y, FOV1z, yp1,zp1
#cam2 = xc2, zc2, FOV2x, FOV2z, xp2,zp2
#cam3 = xc3, yc3, FOV3x, FOV3y, xp3,yp3

a=100
b=100
c=100


yc1 = 100
zc1 = 100
xc2 = -100
zc2 = 100
xc3 = 100
yc3 = 100

yp1 = 320
zp1 = 240
xp2 = 320
zp2 = 240
xp3 = 240
yp3 = 320


FOV1y = 100
FOV1z = 84
FOV2x = 100
FOV2z = 84
FOV3x = 100
FOV3y = 84

FOV = np.array([FOV1y,FOV1z,FOV2x,FOV2z,FOV3x,FOV3y])*0.5/180*np.pi
tanFOV = np.tan(FOV)
cameracoordinates = np.array([yc1,zc1,xc2,zc2,xc3,yc3])
camerapixels = np.array([yp1,zp1,xp2,zp2,xp3,yp3])
tans = cameracoordinates*tanFOV/camerapixels


[ta1,tb1,ta2,tb2,ta3,tb3]=tans   
mx1y1=np.array([[1,-ta2],[ta1,1]])#cam1,2
[x1,y1]=np.linalg.inv(mx1y1).dot(np.array([-b*ta2,a*ta1]))
mx2z1=np.array([[1,ta3],[tb1,1]])#cam1,3
[x2,z1]=np.linalg.inv(mx2z1).dot(np.array([c*ta3,a*tb1]))
my2z2=np.array([[1,tb3],[tb2,1]])#cam2,3
[y2,z2]=np.linalg.inv(my2z2).dot(np.array([c*tb3,b*tb2]))

def real_coordinates(tans,a,b,c):
    [ta1,tb1,ta2,tb2,ta3,tb3]=tans   
    mx1y1=np.array([[1,-ta2],[ta1,1]])#cam1,2
    [x1,y1]=np.linalg.inv(mx1y1).dot(np.array([-b*ta2,a*ta1]))
    mx2z1=np.array([[1,ta3],[tb1,1]])#cam1,3
    [x2,z1]=np.linalg.inv(mx2z1).dot(np.array([c*ta3,a*tb1]))
    my2z2=np.array([[1,tb3],[tb2,1]])#cam2,3
    [y2,z2]=np.linalg.inv(my2z2).dot(np.array([c*tb3,b*tb2]))
    x=0.5*(x1+x2)
    y=0.5*(y1+y2)
    z=0.5*(z1+z2)
    return [x,y,z]
    
    