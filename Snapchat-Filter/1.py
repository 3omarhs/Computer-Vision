import cv2 as cv
import numpy as np


def drawImage(frame, img, x, y, w, h, colorkey):
    resized_img = cv.resize(img, None, fx=w / img.shape[1], fy=h / img.shape[0], interpolation=cv.INTER_NEAREST)

    def check_color(color, mask, offset=20):
        if color[0] >= mask[0] - offset and color[1] >= mask[1] - offset and color[2] >= mask[2] - offset:
            return True
        return False

    for i in range(h):
        for j in range(w):
            if check_color(resized_img[i, j], colorkey) is False:
                frame[y + i, x + j] = resized_img[i, j]


def main():
    face_cascade = cv.CascadeClassifier()
    if not face_cascade.load('haarcascade_frontalface_default.xml'):
        return

    cap = cv.VideoCapture(0)
    img = cv.imread('hair1.png')

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)

        # detect face
        faces = face_cascade.detectMultiScale(frame_gray)

        for (x, y, w, h) in faces:
            drawImage(frame, img, x, y, w, h, np.array([255, 255, 255], dtype=img.dtype))

        cv.imshow('video', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()