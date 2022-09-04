import math
import numpy
import cv2

f1=cv2.imread("111.png")
f2=f1

# B G R

red_color = numpy.int16(f2[:, :, 2])-numpy.int16(f2[:, :, 0])-numpy.int16(f2[:, :, 1])

red_color[red_color>255]=255
red_color[red_color<0]=0

red_color = numpy.uint8(red_color)

red_color[red_color>100]=255
red_color[red_color<=100]=0

print(red_color.shape)

cv2.imshow("my first image", red_color)
cv2.waitKey(0)
