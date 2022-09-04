import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('D:\haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier("D:\haarcascade_smile.xml")

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('D:\haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    cnt = 1
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        smiles = smileCascade.detectMultiScale(gray, 1.8, 15)
        for (x, y, w, h) in smiles:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
            print("Image " + str(cnt) + "Saved")
            path = r'C:\Users\BOSS\Desktop\SmileCapture\images\img' + str(cnt) + '.jpg'
            cv2.imwrite(path, img)
            cnt += 1
            if (cnt >= 2):
                break

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()