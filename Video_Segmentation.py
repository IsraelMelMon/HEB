import numpy as np 
import cv2
from matplotlib import pyplot as plt 


"""
cap = cv2.VideoCapture(0)
gray= cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY) 
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("frame", thresh)
cv2.waitKey(0)
"""
import cv2
 
capture = cv2.VideoCapture(0)
 
while(True):
     
    ret, frame = capture.read() 

    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('video', thresh)
     
    if cv2.waitKey(1) == 27:
        break
 
capture.release()
cv2.destroyAllWindows()