import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# pip install opencv-python
# pip install matplotlib

def resize():
    global image
    h, w = image.shape[:2]
    h = h + int(h*(-0.1))
    w = w + int(w*(-0.1))
    image = cv2.resize(image, (w, h), interpolation= cv2.INTER_LINEAR)
    cv2.imshow('obrazek', image)

def image_canny():
    global image
    b=cv2.blur(image, (cv2.getTrackbarPos('high','obrazek'),cv2.getTrackbarPos('high','obrazek')))
    cv2.imshow('obrazek2', cv2.Canny(b, 55.0, 30.0))
    cv2.imshow('obrazek', b)

def change_color():
    global image
    # color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow('obrazek', color)
    # cv2.imshow('obrazek1', color[:,:,cv2.getTrackbarPos('high','obrazek')])

    high_color = cv2.getTrackbarPos('high','obrazek')
    lower = np.array([0,50,50]) 
    upper = np.array([high_color,220,220]) 
    mask = cv2.inRange(color, lower, upper) 
    cv2.imshow('obrazek', mask)
    res = cv2.bitwise_and(image, image, mask=mask) 
    res = cv2.medianBlur(res, ksize=5) 
    cv2.imshow('obrazek', res)

def change(x):
    global fun
    fun()


image = None
fun = None
files = None

def main():
    global image, fun, files
    files = os.listdir('pliki/')
    image = cv2.imread('pliki/{}'.format(files[0]))
    cv2.imshow('obrazek', image)
    cv2.createTrackbar('high','obrazek',0,255,change)
    while True:
        key = cv2.waitKey()
        if key == ord('-'):
            fun = resize
            fun()
        elif key == ord('q'):
            fun = image_canny
            fun()
        elif key == ord('w'):
            fun = change_color
            fun()
        elif key == 27: # Esc
            cv2.destroyAllWindows()
            break


if __name__=='__main__': 
    main()