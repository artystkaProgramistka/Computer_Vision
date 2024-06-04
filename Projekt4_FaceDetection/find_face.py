import cv2
import numpy as np


fun = None
img = None
i = 0
object_cascade = None

f = 105 # default scale factor for the classifier
n = 5 # minimum neighbors for the classifier
s = 50 # minimum size for the classifier

list_haarcascade = ['haarcascade_eye.xml',
            'haarcascade_eye_tree_eyeglasses.xml',
            'haarcascade_frontalcatface.xml',
            'haarcascade_frontalcatface_extended.xml',
            'haarcascade_frontalface_alt.xml',
            'haarcascade_frontalface_alt2.xml',
            'haarcascade_frontalface_alt_tree.xml',
            'haarcascade_frontalface_default.xml',
            'haarcascade_fullbody.xml',
            'haarcascade_lefteye_2splits.xml',
            'haarcascade_license_plate_rus_16stages.xml',
            'haarcascade_lowerbody.xml',
            'haarcascade_profileface.xml',
            'haarcascade_righteye_2splits.xml',
            'haarcascade_russian_plate_number.xml',
            'haarcascade_smile.xml',
            'haarcascade_upperbody.xml']
# The Haar cascade XML files are part of the OpenCV library

def set_object():
    global i, object_cascade, list_haarcascade
    object_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + list_haarcascade[i])
    print(list_haarcascade[i])
    i += 1
    if i >= len(list_haarcascade):
        i=0

def bar_fun(x):
    global f, n, s
    f = cv2.getTrackbarPos('scaleFactor', 'bar')
    n = cv2.getTrackbarPos('minNeighbors', 'bar')
    s = cv2.getTrackbarPos('minSize', 'bar')


def main():
    global swich_lib, fun, img, object_cascade, f, n, s
    cap = cv2.VideoCapture('AmericanPsycho_cut.mp4')
    set_object()
    ret, frame = cap.read()
    cv2.imshow('bar', frame)
    cv2.createTrackbar('scaleFactor', 'bar', 101, 1000, bar_fun)
    cv2.createTrackbar('minNeighbors', 'bar', 1, 500, bar_fun)
    cv2.createTrackbar('minSize', 'bar', 1, 500, bar_fun)
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = object_cascade.detectMultiScale(
                gray, scaleFactor=f / 100, minNeighbors=n, minSize=(s, s))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                img = frame[y:y + h, x:x + w]
                img = cv2.resize(img, (200, 200),
                                 interpolation=cv2.INTER_LINEAR)
                cv2.imshow('twarz', img)

            cv2.imshow('bar', frame)

        key = cv2.waitKey(1)
        if key == ord('n'):
            set_object()
        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
