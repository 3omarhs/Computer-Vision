import math
import numpy
import cv2

f1=cv2.imread("1.jpg")
f11=cv2.imread("2.jpg")
f2=f1

#f2[:,:,0]=0
#f2[:,:,1]=0
#f2[:,:,2]=0
#B,G,R

#cv2.resize(f11 (385,385))

#f3=cv2.subtract(f1,f11)
f3=f1-f11
print(f2[:,:,0])

cv2.imshow("my first image",f3)
cv2.waitKey(0)

