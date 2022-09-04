# import cv2
#
# def back():
#     1
#
# img = cv2.imread(f'photos/logo.jpg') #, cv2.IMREAD_UNCHANGED)
# img = cv2.resize(img, (350, 350), interpolation= cv2.INTER_AREA)
# cv2.createButton("Back",back)
# cv2.imshow("Image..", img)
# cv2.waitKey(0)
#


# '''
import cv2
from tkinter import Label, Tk, Button, ttk
from PIL import ImageTk, Image
import os

win = Tk()
win.geometry("600x600")

filters = "RGB", "GRAY", "HSV", "FHSV", "HLS"
com_box = ttk.Combobox(win, values=filters)
com_box.current(0)
com_box.place(x=50, y=500, width=100)


def change_filter(img):
    if com_box.get() == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if com_box.get() == "GRAY":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if com_box.get() == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if com_box.get() == "FHSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    if com_box.get() == "HLS":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    return img


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


def switch(i):
    label = Label(win, bg="black")
    img = cv2.imread(list2[i])
    img = change_filter(img)
    to_display(img, label, 150, 20, 300, 400)


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


#     win.after(700,coundown)
# coundown()

right = Button(win, text="▶", bg="gray", fg="white", command=counup).place(x=330, y=500, width=40)
left = Button(win, text="◀", bg="gray", fg="white", command=coundown).place(x=230, y=500, width=40)

win.mainloop()
# '''
