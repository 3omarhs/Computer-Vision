from cvzone.SerialModule import SerialObject

arduino = SerialObject('COM5')

while True:
    myData = arduino.getData()
    print(myData[0])