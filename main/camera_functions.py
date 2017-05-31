import cv2
import time
import math
import numpy as np
from threading import Thread


def frame_loc(vc, first_frame=None, imshow=None, thres=60):
    """
    Takes the videoCapture object, first frame and cam_num as the input,
    returns the drone location.
    """
    rval, frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    frameDelta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frameDelta, thres, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    x, y, w, h = cv2.boundingRect(thresh)
    if imshow != None:
        frame = cv2.rectangle(frameDelta, (int(x+w/2-10), int(y+h/2-10)),
                              (int(x+w/2+10), int(y+h/2+10)), (255, 255, 255), 2)
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


def get_coordinates(cam1_tan, cam2_tan, cam3_tan, a=0.815, b=0.815, c=0.815):
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


def threaded_loop_test(vc, first_frame, imshow=None):
    t = time.time()
    for i in range(1000):
        x, y = frame_loc(vc, first_frame, imshow)
        # print(x, " ", y)
        key = cv2.waitKey(10)
        if key == 27:  # exit on ESC
            break
    print(time.time()-t)


def threaded_loop(coordinates, vc0, vc1, vc2, first_frame0, first_frame1,
                  first_frame2, imshow0=None,
                  imshow1=None, imshow2=None):
    while True:
        x0, y0 = frame_loc(vc0, first_frame0, imshow0, 30)
        x1, y1 = frame_loc(vc1, first_frame1, imshow1, 30)
        x2, y2 = frame_loc(vc2, first_frame2, imshow2, 30)

        coordinates[:] = get_coordinates(
            get_angle(x0, y0), get_angle(x1, y1), get_angle(x2, y2))[:]
        # print(coordinates+[x0, y0, x1, y1, x2, y2])
        key = cv2.waitKey(10)
        if key == 27:  # exit on ESC
            cv2.VideoCapture(1).release()
            cv2.VideoCapture(2).release()
            cv2.VideoCapture(3).release()
            break


def simplified_loop(coordinates,read_failed):
    vc0, vc1, vc2, first_frame0, first_frame1, first_frame2 = Init()
    while True:
        colored_Cam(coordinates, read_failed, vc0, vc1, vc2,
            first_frame0, first_frame1, first_frame2)
        key = cv2.waitKey(10)
        if key == 27:  # exit on ESC
            cv2.VideoCapture(1).release()
            cv2.VideoCapture(2).release()
            cv2.VideoCapture(3).release()
            break


def Init():
    cv2.VideoCapture(1).release()
    cv2.VideoCapture(2).release()
    cv2.VideoCapture(3).release()

    # Cam 0
    vc0 = cv2.VideoCapture(1)
    vc0.set(3, 640)
    vc0.set(4, 240)
    # vc0.set(15, -7.2)  # exposure
    # Cam 1
    vc1 = cv2.VideoCapture(2)
    vc1.set(3, 640)
    vc1.set(4, 240)
    # vc1.set(15, -7.2)  # exposure
    # Cam 2
    vc2 = cv2.VideoCapture(3)
    vc2.set(3, 640)
    vc2.set(4, 240)
    # vc2.set(15, -6.5)  # exposure
    for i in range(5):
        rval0, frame0 = vc0.read()
        rval1, frame1 = vc1.read()
        rval2, frame2 = vc2.read()
    assert vc0.isOpened(), "can't find camera 0"
    assert vc1.isOpened(), "can't find camera 1"
    assert vc2.isOpened(), "can't find camera 2"
    first_frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    first_frame0 = cv2.GaussianBlur(first_frame0, (21, 21), 0)
    first_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    first_frame1 = cv2.GaussianBlur(first_frame1, (21, 21), 0)
    first_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    first_frame2 = cv2.GaussianBlur(first_frame2, (21, 21), 0)

    return vc0, vc1, vc2, first_frame0, first_frame1, first_frame2


def Cam(coordinates, read_failed, vc0, vc1, vc2,
        first_frame0, first_frame1, first_frame2,
        testing=True):
    # high level function, takes in the coordinates, read_failed list,
    # VideoCapture objects and their first frames, mutates these lists
    if testing:
        imshow0, imshow1, imshow2 = "cam x", "cam y", "cam z"
    else:
        imshow0 = imshow1 = imshow2 = None
    x0, y0 = frame_loc(vc0, first_frame0, imshow0, 30)
    x1, y1 = frame_loc(vc1, first_frame1, imshow1, 40)
    x2, y2 = frame_loc(vc2, first_frame2, imshow2, 40)

    coordinates[:] = get_coordinates(
        get_angle(x0, y0), get_angle(x1, y1), get_angle(x2, y2))[:]
    # print([x0, y0, x1, y1, x2, y2])
    if x0*x1*x2 != 0 or y0*y1*y2 != 0:  # captured by all 3 cams
        read_failed[0] = 0
        print("found drone")
    else:
        read_failed[0] = 1

def callback(value):
    pass

def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)
 
    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255
 
        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)

def get_trackbar_values(range_filter):
    values = []
 
    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)
    return values

def colored_Cam(coordinates, read_failed, vc0, vc1, vc2,
        first_frame0, first_frame1, first_frame2,
        testing=True):
    setup_trackbars('RGB')
    rval, frame = vc0.read()
    frame_to_thresh = frame.copy()
    v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values('RGB')
    thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
 
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 3, (0, 0, 255), -1)
            cv2.putText(frame,"centroid", (center[0]+10,center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)
            cv2.putText(frame,"("+str(center[0])+","+str(center[1])+")", (center[0]+10,center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)

if __name__ == '__main__':
    coordinates = [0, 0, 0]
    read_failed = [1]
    # vc0, vc1, vc2, first_frame0, first_frame1, first_frame2 = Init()

    # Thread(target=threaded_loop_test, args=(vc0, first_frame0, "0",)).start()
    # Thread(target=threaded_loop_test, args=(vc1, first_frame1, "1",)).start()

    # camera_Thread = Thread(
    #     target=threaded_loop, args=(coordinates,
    #                                 vc0, vc1, vc2, first_frame0, first_frame1,
    #                                 first_frame2, "0", "1", "2",))
    camera_Thread = Thread(target=simplified_loop, args=(coordinates, read_failed))
    camera_Thread.start()
    while 1:
        # print(coordinates)
        time.sleep(1)
    # threaded_loop(vc0, vc1, vc2, first_frame0, first_frame1, first_frame2, "0", "1", "2",)
