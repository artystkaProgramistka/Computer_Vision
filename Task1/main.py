import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# pip install opencv-python
# pip install matplotlib

def load_next_image():
    global image_index, image, files, original_image
    image_index = (image_index + 1) % len(files)  # Loop back to the first image
    image_path = os.path.join('pliki', files[image_index])
    image = cv2.imread(image_path)
    original_image = image.copy()  # Update the original image
    cv2.imshow('obrazek', image)

def resize(new_scale=None):
    global image, original_image, scale
    # If a new scale is provided, update the global scale
    if new_scale is not None:
        scale = min(max(10, new_scale), 200)  # Limit the scale to be between 10 and 200
        # Update the trackbar position to match the new scale
        cv2.setTrackbarPos('Image size', 'obrazek', int(scale))

    # Compute new dimensions based on the global scale
    h, w = original_image.shape[:2]
    new_h = int(h * (scale / 100.0))
    new_w = int(w * (scale / 100.0))

    # Resize the image and show it
    image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
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
    if fun is not None:
        fun()

image = None
fun = resize
files = None
image_index = 0
original_image = None
scale = 0.1

def main():
    global image, fun, files, image_index, original_image, scale
    files = [f for f in os.listdir('pliki/') if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if not files:
        print("No image found in 'pliki' directory.")
        return
    image = cv2.imread('pliki/{}'.format(files[image_index]))
    original_image = image.copy()
    cv2.imshow('obrazek', image)
    cv2.createTrackbar('Image size', 'obrazek', 100, 200, resize)
    while True:
        key = cv2.waitKey()
        # changing the image size
        if key == ord('-'):
            new_scale = scale - 10
            resize(new_scale)
        elif key == ord('+'):
            new_scale = scale + 10
            resize(new_scale)
        elif key == ord('q'):
            fun = image_canny
            fun()
        elif key == ord('w'):
            fun = change_color
            fun()
        # iterating through images
        elif key == ord('n'):
            load_next_image()
        elif key == 27: # Esc
            cv2.destroyAllWindows()
            break


if __name__=='__main__':
    main()