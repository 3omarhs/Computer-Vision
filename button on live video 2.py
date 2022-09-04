from tkinter import *
from PIL import ImageTk, Image
import cv2


root = Tk()
root.geometry("644x550")
# Create a frame
app = Frame(root, bg="white")
app.grid()
# Create a label in the frame
lmain = Label(app)
lmain.grid()

# Capture from camera
cap = cv2.VideoCapture(0)


def counup():
    1


# previous img
def coundown():
    1

# function for video streaming
def video_stream():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream)


video_stream()
santabeard =Button(root, text="s_beard", bg="gray", fg="white", command=counup).place(x=50, y=500, width=50)
santahat =  Button(root, text="s_hat", bg="gray", fg="white", command=coundown).place(x=130, y=500, width=50)
clownhair = Button(root, text="c_hair", bg="gray", fg="white", command=counup).place(x=210, y=500, width=50)
clownnose = Button(root, text="c_nose", bg="gray", fg="white", command=counup).place(x=290, y=500, width=50)
mustache2 = Button(root, text="musta..", bg="gray", fg="white", command=counup).place(x=370, y=500, width=50)
hat =       Button(root, text="hat", bg="gray", fg="white", command=counup).place(x=450, y=500, width=50)
glasses =   Button(root, text="glasses", bg="gray", fg="white", command=counup).place(x=530, y=500, width=50)
root.mainloop()