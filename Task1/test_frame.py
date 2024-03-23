import cv2
import numpy as np
import os

# Define the resize function
def resize(img, s):
    h, w = img.shape[:2]
    h = h + int(h * s)
    w = w + int(w * s)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

# Define the normalization function to keep the larger dimension at 800px
def norm_size(img):
    h, w = img.shape[:2]
    if h > w:
        if h > 800:
            s = (1 - (800 / h)) * (-1)
            img = resize(img, s)
    else:
        if w > 800:
            s = (1 - (800 / w)) * (-1)
            img = resize(img, s)
    return img


def hsv_range():
    # Convert the HSV colorspace
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    # utwórz maskę kolorów jakie znajdują się na piłce przy pomocy operacji binarnej
    red_mask = cv2.bitwise_or(mask1, mask2)
    cv2.imshow('obrazek', red_mask)


def hsv_bitwais():
    low_color = cv2.getTrackbarPos('low hue', 'obrazek')
    high_color = cv2.getTrackbarPos('high hue', 'obrazek')
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 0, 0])
    upper = np.array([high_color, 255, 255])
    # utwórz maskę kolorów jakie znajdują się na piłce przy pomocy operacji binarnej.
    mask = cv2.inRange(hsv_frame, lower, upper)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('obrazek', res)


def hsv_median():
    ksize = cv2.getTrackbarPos('kernel size', 'obrazek')
    # Ensure ksize is odd and at least 3
    while ksize % 2 == 0 or ksize < 3:
        ksize += 1
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    res = cv2.bitwise_or(frame, frame, mask=red_mask)
    res = cv2.medianBlur(res, ksize=ksize)  # Apply median blur with the corrected ksize
    cv2.imshow('obrazek', res)


# Popraw jakość obrazu (usuń szum) poprzez operacje morfologiczne
def morphology():
    low_color = cv2.getTrackbarPos('low hue', 'obrazek')
    high_color = cv2.getTrackbarPos('high hue', 'obrazek')
    ksize = cv2.getTrackbarPos('kernel size', 'obrazek')
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    # utwórz maskę kolorów jakie znajdują się na piłce przy pomocy operacji binarnej.
    mask = cv2.inRange(hsv_frame, lower, upper)
    kernel = np.ones((ksize, ksize), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('obrazek', mask_without_noise)


def morphology2():
    low_color = cv2.getTrackbarPos('low hue', 'obrazek')
    high_color = cv2.getTrackbarPos('high hue', 'obrazek')
    ksize = cv2.getTrackbarPos('kernel size', 'obrazek')
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    # utwórz maskę kolorów jakie znajdują się na piłce przy pomocy operacji binarnej.
    mask = cv2.inRange(hsv_frame, lower, upper)
    kernel = np.ones((7, 7), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((ksize, ksize), np.uint8)
    mask_closed = cv2.morphologyEx(mask_without_noise, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('obrazek', mask_closed)


def marker():
    low_color = cv2.getTrackbarPos('low hue', 'obrazek')
    high_color = cv2.getTrackbarPos('high hue', 'obrazek')
    # Podczas śledzenia zmień format obrazu na HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 0, 0])
    upper = np.array([high_color, 255, 255])
    # utwórz maskę kolorów jakie znajdują się na piłce przy pomocy operacji binarnej
    mask = cv2.inRange(hsv_frame, lower, upper)
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    M = cv2.moments(contours[0])
    print(M)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    image_marker = frame.copy()
    cv2.drawMarker(image_marker, (int(cx), int(cy)), color=(
        0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
    cv2.imshow('obrazek', image_marker)


def change_h(x):
    global fun
    if fun is not None:
        fun()


frame = None
fun = None
files = None

def main():
    global frame, fun, files

    # Open the video
    video = cv2.VideoCapture('movingball.mp4')

    # Go directly to the 50th frame
    frame_index = 49
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Read the 50th frame
    success, frame = video.read()

    # Check if the 50th frame was successfully read
    if success:
        # Resize the frame for display
        frame = norm_size(frame)
        # Display the frame
        cv2.imshow('50th Frame', frame)
        cv2.waitKey(0)  # Wait for any key to be pressed
        cv2.destroyAllWindows()
    else:
        print("Could not read the 50th frame.")

    # Release the video capture object
    video.release()

    nimg = frame.copy()
    cv2.imshow('obrazek', frame)
    cv2.createTrackbar('low hue', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('high hue', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('kernel size', 'obrazek', 5, 50, change_h)

    while True:
        key = cv2.waitKey()
    # ----------------zmiana rozmiaru---------------
        if key == ord('-'):
            frame = resize(frame, -0.1)
            nimg = frame.copy()
            cv2.imshow('obrazek', frame)
        elif key == ord('+'):
            frame = resize(frame, 0.1)
            nimg = frame.copy()
            cv2.imshow('obrazek', frame)
        elif key == ord('='):
            cv2.imshow('obrazek', frame)
            nimg = frame.copy()
    # ----------------kolory------------------------
        elif key == ord('q'):
            cv2.imshow('obrazek', cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        elif key == ord('w'):
            nimg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
            cv2.imshow('obrazek', nimg[:, :, 0])
        elif key == ord('x'):
            # s = nasycene
            cv2.imshow('obrazek', nimg[:, :, 1])
        elif key == ord('c'):
            # v = wartość
            cv2.imshow('obrazek', nimg[:, :, 2])
    # ----------------filtry
        elif key == ord('a'):
            cv2.imshow('obrazek', cv2.Canny(frame, 55.0, 30.0))
        elif key == ord('s'):
            cv2.imshow('obrazek', cv2.blur(frame, (7, 7)))
        elif key == ord('d'):
            b = cv2.blur(frame, (7, 7))
            cv2.imshow('obrazek', cv2.Canny(b, 55.0, 30.0))
        elif key == ord('f'):
            morphology()
            fun = morphology
        elif key == ord('g'):
            morphology2()
            fun = morphology
        elif key == ord('h'):
            marker()
            fun = marker
        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
