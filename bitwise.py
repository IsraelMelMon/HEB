
"""
import numpy as np 
import cv2
from matplotlib import pyplot as plt 

mask= cv2.imread('2020-02-20_08-38-40.jpg')
result= cv2.bitwise_and(frame, frame, mask=mask)

ret, thresh = cv2.threshold(result,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("frame", thresh)
cv2.waitKey(0)



cv2.bitwise_and()
"""

def execute_HSV(proxy,obj):

	say("hsv ..")

	try: img=obj.sourceObject.Proxy.img.copy()
	except: img=cv2.imread(__dir__+'/icons/freek.png')

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	lower=np.array([obj.valueColor-obj.deltaColor,0,0])
	upper=np.array([obj.valueColor+obj.deltaColor,255,255])
	mask = cv2.inRange(hsv, lower, upper)

	res = cv2.bitwise_and(hsv,hsv, mask= mask)

	obj.Proxy.img=res 