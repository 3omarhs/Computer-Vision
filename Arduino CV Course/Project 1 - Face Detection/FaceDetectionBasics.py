import cv2
from cvzone.FaceDetectionModule import FaceDetector
cap = cv2.VideoCapture(0)


detector = FaceDetector()  # according to accuracy take face or not: detector = FaceDetector(minDetectionCon=0.9)

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
