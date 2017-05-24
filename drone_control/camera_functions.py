import cv2
import time


def get_loc(vc, first_frame=None, cam_num=0, imshow = None):
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
    return vc, first_frame, x+w/2, y+h/2

def get_angle(x, y, w=640, h=480):
    """
    From the x,y location read from the camera, get the horizontal angle
    alpha and vertical angle beta.
    """
    ###insert function here
    return alpha, beta


if __name__ == '__main__':
    vc = cv2.VideoCapture(0)
    vc.set(3, 640)
    vc.set(4, 480)
    vc.set(16, -6.0)#exposure
    assert vc.isOpened(), "can't find camera"
    rval, frame = vc.read()
    first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)
    cv2.namedWindow("preview")
    t1 = time.time()
    for i in range(200):
        vc, first_frame, x, y = get_loc(vc, first_frame, 0,"preview")
        print(x, " ", y)
        key = cv2.waitKey(10)
        if key == 27:  # exit on ESC
            break
    print(time.time()-t1)
