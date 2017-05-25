import cv2
import time
import math
import numpy as np
from threading import Thread


def frame_loc(vc, first_frame=None, cam_num=0, imshow=None):
    """
    Takes the videoCapture object, first frame and cam_num as the input,
    returns the drone location.
    """
    rval, frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    frameDelta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    x, y, w, h = cv2.boundingRect(thresh)
    if imshow != None:
        cv2.imshow(imshow, frame)
    return x+w/2, y+h/2


def get_angle(x, y, w=640, h=480):
    """
    From the x,y location read from the camera, get tangent of the
    horizontal angle alpha and vertical angle beta.
    """
    x = x-w/2
    y = h/2-y
    ta = x*math.tan(math.radians(30))/(w/2)
    tb = y*math.tan(math.radians(25))/(h/2)
    return [ta, tb]


def get_coordinates(cam1_tan, cam2_tan, cam3_tan, a, b, c):
    # tans ==> array of 6 tangents
    [ta1, tb1], [ta2, tb2], [ta3, tb3] = cam1_tan, cam2_tan, cam3_tan
    mx1y1 = np.array([[1, -ta2], [ta1, 1]])  # cam1,2
    [x1, y1] = np.linalg.inv(mx1y1).dot(np.array([-b*ta2, a*ta1]))
    mx2z1 = np.array([[1, ta3], [tb1, 1]])  # cam1,3
    [x2, z1] = np.linalg.inv(mx2z1).dot(np.array([c*ta3, a*tb1]))
    my2z2 = np.array([[1, tb3], [tb2, 1]])  # cam2,3
    [y2, z2] = np.linalg.inv(my2z2).dot(np.array([c*tb3, b*tb2]))
    x = 0.5*(x1+x2)
    y = 0.5*(y1+y2)
    z = 0.5*(z1+z2)
    return np.array([x, y, z])


def threaded_loop(vc, first_frame, imshow):
    t = time.time()
    for i in range(200):
        x, y = frame_loc(vc, first_frame, 0, "preview")
        print(x, " ", y)
        key = cv2.waitKey(10)
        if key == 27:  # exit on ESC
            break
    print(time.time()-t)

if __name__ == '__main__':
    vc = cv2.VideoCapture(0)
    vc.set(3, 640)
    vc.set(4, 480)
    vc.set(16, -6.0)  # exposure
    assert vc.isOpened(), "can't find camera"
    rval, frame = vc.read()
    first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)
    # cv2.namedWindow("preview")
    Thread(target=threaded_loop, args=(vc, first_frame, "preview",)).start()
    while 1:
        time.sleep(1)
