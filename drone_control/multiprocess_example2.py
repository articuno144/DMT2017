import numpy as np
import cv2
from multiprocessing import Process

class Eye():
  def __init__(self, n):
    self.n = n
    self.cap = cv2.VideoCapture(self.n)

  def cam_eye(self):
    ret, frame = self.cap.read()
    cv2.imshow("{0}".format(self.n),frame) 

  def close(self):
    self.cap.release()

if __name__ == "__main__":
  left =Eye(0)
  Process(target=left.cam_eye, args=(0,)).start()
  while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left.close()

