import cv2
from tkinter import Label, Tk, Button, ttk
from PIL import ImageTk, Image
import os



cap = cv2.VideoCapture()
# The device number might be 0 or 1 depending on the device and the webcam
cap.open(0, cv2.CAP_DSHOW)
while(True):
    ret, frame = cap.read()
    # cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    win = Tk()
    win.geometry("600x600")

    path = os.getcwd()
    # myList = os.listdir(path)
    myPath = 'D:/Backup/PycharmProjects/datasets/10photos'
    myList = os.listdir(myPath)
    list2 = []
    for _, imgs in enumerate(myList):
        img = imgs.split(".")
        if img[1] == "jpg" or img[1] == "png":
            imgg = 'D:/Backup/PycharmProjects/datasets/10photos/' + imgs
            # list2.append(imgs)
            list2.append(imgg)


    # print(len(list2))

    # print(myList)
    def to_display(img, label, x, y, w, h):
        image = Image.fromarray(img)
        image = image.resize((w, h), Image.ANTIALIAS)
        pic = ImageTk.PhotoImage(image)
        label.configure(image=pic)
        label.image = pic
        label.place(x=x, y=y)

    # to_display(imgg, 'label', 150, 20, 300, 400)

    def switch(i):
        label = Label(win, bg="black")
        img = cv2.imread(list2[i])
        # img = change_filter(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        to_display(img, label, 150, 20, 300, 400)


    def switch():
        label = Label(win, bg="black")
        # img = cv2.imread(list2[i])
        ret, frame = cap.read()
        cv2.waitKey(1)
        img = frame
        # img = change_filter(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        to_display(img, label, 150, 20, 300, 400)


    switch()
    count = 0


    # next img
    def counup():
        global count
        count += 1
        if count > len(list2) - 1:
            count = 0
        switch(count)


    # previous img
    def coundown():
        global count
        count -= 1
        if count < 0:
            count = len(list2) - 1
        switch(count)



    # right = Button(win, text="▶", bg="gray", fg="white", command=counup).place(x=330, y=500, width=40)
    # left = Button(win, text="◀", bg="gray", fg="white", command=coundown).place(x=230, y=500, width=40)

    win.mainloop()

cap.release()
cv2.destroyAllWindows()