import cv2

cv2.namedWindow("vc1")
vc1 = cv2.VideoCapture(0)
vc1.set(3, 640)
vc1.set(4, 480)
vc1.set(6, 60)
if vc1.isOpened():  # try to get the first frame
    rval1, frame1 = vc1.read()
else:
    rval1 = False

cv2.namedWindow("vc2")
vc2 = cv2.VideoCapture(1)
vc2.set(3, 640)
vc2.set(4, 480)
vc2.set(6, 60)
if vc2.isOpened():  # try to get the first frame
    rval2, frame2 = vc2.read()
else:
    rval2 = False

def get_loc(frame, k):
    ########################### TO DO #################################
    #print the location of the darkest k regions

while rval1:
    cv2.imshow("vc1", frame1)
    rval1, frame1 = vc1.read()    
    cv2.imshow("vc2", frame2)
    rval2, frame2 = vc2.read()
    print(get_loc(frame1, 1))
    print(get_loc(frame2, 1))
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
cv2.destroyWindow("vc2")
cv2.destroyWindow("vc1")