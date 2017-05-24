import cv2

cv2.namedWindow("vc1")
vc1 = cv2.VideoCapture(0)
vc1.set(3, 640)
vc1.set(4, 480)
vc1.set(6, 60)
if vc1.isOpened():  # try to get the first frame
    rval1, frame1 = vc1.read()
    firstframe1 = None
else:
    rval1 = False

cv2.namedWindow("vc2")
vc2 = cv2.VideoCapture(1)
vc2.set(3, 640)
vc2.set(4, 480)
vc2.set(6, 60)
vc2.set(16, -5.0)#exposure
if vc2.isOpened():  # try to get the first frame
    rval2, frame2 = vc2.read()
    firstframe2 = None
else:
    rval2 = False


def get_loc(firstframe, gray):
    # print the location of the darkest k regions frame 480*640*3
    frameDelta = cv2.absdiff(firstframe, gray)
    thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    x, y, w, h = cv2.boundingRect(thresh)
    # frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    #cv2.imshow("frame", frame)
    #cv2.imshow("Thresh", thresh)
    #cv2.imshow("frame2", frameDelta)
    return x, y, w, h

while rval1:
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    if firstframe1 is None:
        firstframe1 = gray1
        continue
    if firstframe2 is None:
        firstframe2 = gray2
        continue
    x1, y1, w1, h1 = get_loc(firstframe1, gray1)
    x2, y2, w2, h2 = get_loc(firstframe2, gray2)
    #print(get_loc(frame1, 1))
    #print(get_loc(frame2, 1))
    frame1 = cv2.rectangle(frame1, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
    frame2 = cv2.rectangle(frame2, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 2)
    cv2.imshow("vc1", frame1)
    rval1, frame1 = vc1.read()
    cv2.imshow("vc2", frame2)
    rval2, frame2 = vc2.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
cv2.destroyWindow("vc2")
cv2.destroyWindow("vc1")
