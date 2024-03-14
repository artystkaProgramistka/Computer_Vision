import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def uploud(i):
    global files
    i=i-ord('0')
    image = norm_size(cv2.imread('pliki/{}'.format(files[i])))
    cv2.imshow('obrazek', image)
    return image

def resize(img, s):
    h, w = img.shape[:2]
    h = h + int(h*s)
    w = w + int(w*s)
    return cv2.resize(img, (w, h), interpolation= cv2.INTER_LINEAR)

def norm_size(img):
    h, w = img.shape[:2]
    if h > w:
        if h > 800:
            s = (1 - (800/h)) * (-1)
            img = resize(img, s)
    else:
        if w > 800:
            s = (1 - (800/w)) * (-1)
            img = resize(img, s)
    return img

def hsv_range():
    low_color = cv2.getTrackbarPos('low','obrazek')
    high_color = cv2.getTrackbarPos('high','obrazek')
    # Convert the HSV colorspace 
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    # Threshold the HSV image to get only blue color 
    lower = np.array([low_color,100,100]) 
    upper = np.array([high_color,255,255]) 
    mask = cv2.inRange(hsv_frame, lower, upper) 
    cv2.imshow('obrazek', mask)

def hsv_bitwais():
    low_color = cv2.getTrackbarPos('low','obrazek')
    high_color = cv2.getTrackbarPos('high','obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    lower = np.array([low_color,100,100]) 
    upper = np.array([high_color,255,255]) 
    mask = cv2.inRange(hsv_frame, lower, upper) 
    # Bitwise-AND mask and original image 
    res = cv2.bitwise_and(image, image, mask=mask) 
    cv2.imshow('obrazek', res)

def hsv_median():
    low_color = cv2.getTrackbarPos('low','obrazek')
    high_color = cv2.getTrackbarPos('high','obrazek')
    ksize = cv2.getTrackbarPos('ksize','obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    lower = np.array([low_color,100,100]) 
    upper = np.array([high_color,255,255]) 
    mask = cv2.inRange(hsv_frame, lower, upper) 
    res = cv2.bitwise_and(image, image, mask=mask) 
    res = cv2.medianBlur(res, ksize=ksize) 
    cv2.imshow('obrazek', res)

def change_h(x):
    global fun
    if fun is not None:
        fun()

image = None
fun = None
files = None

def main():
    global image, fun, files
    files = os.listdir('pliki')
    num_image = ord('0')
    image = norm_size(cv2.imread('pliki/{}'.format(files[0])))
    nimg = image.copy()
    cv2.imshow('obrazek', image)
    cv2.createTrackbar('low','obrazek',0,255,change_h)
    cv2.createTrackbar('high','obrazek',0,255,change_h)
    cv2.createTrackbar('ksize','obrazek',5,50,change_h)


    while True:
        key = cv2.waitKey()
    #-----------wybor obrazka----------------
        if key >= ord('0') and key <= ord('9'):
            num_image = key
            image = uploud(key)
            nimg=image.copy()
    #----------------zmiana rozmiaru---------------
        elif key == ord('-'):
            image = resize(image, -0.1)
            nimg=image.copy()
            cv2.imshow('obrazek', image)
        elif key == ord('+'):
            image = resize(image, 0.1)
            nimg=image.copy()
            cv2.imshow('obrazek', image)
        elif key == ord('='):
            image = uploud(num_image)
            cv2.imshow('obrazek', image)
            nimg=image.copy()
    #----------------kolory------------------------
        elif key == ord('q'):
            cv2.imshow('obrazek', cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        elif key == ord('w'):
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            cv2.imshow('obrazek', nimg)
        elif key == ord('e'):
            hsv_range()
            fun = hsv_range
        elif key == ord('r'):
            hsv_bitwais()
            fun = hsv_bitwais
        elif key == ord('t'):
            hsv_median()
            fun = hsv_median
        elif key == ord('z'):
            # h = barwa
            cv2.imshow('obrazek', nimg[:,:,0])
        elif key == ord('x'):
            # s = nasycene
            cv2.imshow('obrazek', nimg[:,:,1])
        elif key == ord('c'):
            # v = wartość
            cv2.imshow('obrazek', nimg[:,:,2])
    #----------------filtry
        elif key == ord('a'):
            cv2.imshow('obrazek', cv2.Canny(image, 55.0, 30.0))
        elif key == ord('s'):
            cv2.imshow('obrazek', cv2.blur(image, (7,7)))
        elif key == ord('d'):
            b=cv2.blur(image, (7,7))
            cv2.imshow('obrazek', cv2.Canny(b, 55.0, 30.0))
        elif key == 27:
            cv2.destroyAllWindows()
            break



if __name__=='__main__': 
    main()