import numpy as np 
import cv2
from matplotlib import pyplot as plt 
from scipy import integrate as inte 

"""
cap = cv2.VideoCapture(0)
gray= cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY) 
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("frame", thresh)
cv2.waitKey(0)
"""
import cv2
 
capture = cv2.VideoCapture(0)
capture.set(15, 25)
 
'''
Histogram stuff
'''
bins = 400


fig, ax = plt.subplots()

ax.set_title('Histogram (rgb)')
ax.set_xlabel('Bin')
ax.set_ylabel('Frequency')
lw = 3
alpha = 0.5
lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw, label='intensity')
ax.set_xlim(0, bins-1)
ax.set_ylim(0, 1)
ax.legend()
plt.ion()
plt.show()

lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha, label='Red')
lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha, label='Green')
lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha, label='Blue')

while(True):
     
    ret, frame = capture.read() 
    
    NewImg = frame.copy()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    ret, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    
    #ret, thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
     #       cv2.THRESH_BINARY,3,2)
    #cv2.imshow('video', thresh)
    scale_percent = 30 # percent of original size
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(thresh, dim, interpolation = cv2.INTER_AREA)
    resized2 = cv2.resize(NewImg,dim,interpolation=cv2.INTER_AREA)
    #cv2.imshow('mask', resized)

    bit_and = cv2.bitwise_and(resized2, resized2, mask=resized)
    #cv2.imshow('mask', resized2)

    #cv2.bitwise NewImg
    cv2.imshow('video', bit_and)
    #cv2.imshow('video2', resized)

    '''
    Histogram stuff
    '''

    numPixels = np.prod(bit_and.shape[:2])
  #  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #  cv2.imshow('Grayscale', gray)
    histogram = cv2.calcHist([bit_and], [0], None, [bins], [0, 255]) / numPixels


    (b, g, r) = cv2.split(frame)
    histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / numPixels
    histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
    histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
    lineR.set_ydata(histogramR)
    lineG.set_ydata(histogramG)
    lineB.set_ydata(histogramB)
    #print(histogram)
    #bin_width = bins[1] - bins[0]
    #integral = sum(bit_and)
    #print(integral)
    
    #lineGray.set_ydata(histogram)
    fig.canvas.draw()
    #z = inte.quad(0, bins)
    if cv2.waitKey(1) == 27:
        break

    elif cv2.waitKey(0) == 99:
        cv2.imwrite("zane4_2.png", NewImg)

   # else:
   #     print("whut")

 
capture.release()
cv2.destroyAllWindows()