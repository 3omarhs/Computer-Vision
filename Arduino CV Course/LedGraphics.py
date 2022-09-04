from cvzone.SerialModule import SerialObject
from time import sleep
import cv2

arduino = SerialObject('COM5')

# imgLedOn = cv2.imread("../Arduino CV Course/Resources/LedOn.jpg")
# imgLedOff = cv2.imread("../Arduino CV Course/Resources/LedOff.jpg")
imgLedOn = cv2.imread("../Arduino CV Course/Resources/Pin13On.jpg")
imgLedOff = cv2.imread("../Arduino CV Course/Resources/Pin13Off.jpg")

while True:
    arduino.sendData([1])
    cv2.imshow("Image", imgLedOn)
    cv2.waitKey(3000)
    arduino.sendData([0])
    cv2.imshow("Image", imgLedOff)
    cv2.waitKey(1000)