import math
import numpy
import cv2

f1=cv2.imread("1.jpg")
f2=f1

f3=cv2.cvtColor(f1,cv2.COLOR_BGR2GRAY)

print(f3.shape)
print(f3)
f4=f3
for i in range(0,385):#512
    for j in range(0,385):#512
        if(f4[i,j]>=120):
            f4[i,j]=0
        else:
            f4[i,j]=1

cv2.imshow("my first image", f4)
cv2.waitKey(0)
