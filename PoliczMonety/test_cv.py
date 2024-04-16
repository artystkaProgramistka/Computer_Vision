import cv2
import numpy as np
import os
import math
from PIL import Image

# pip install opencv-python

def draw_cross(img, center, size=20, color=(0, 255, 0), thickness=2):
    # Draw a cross at the specified center point
    cv2.line(img, (center[0] - size, center[1]), (center[0] + size, center[1]), color, thickness)
    cv2.line(img, (center[0], center[1] - size), (center[0], center[1] + size), color, thickness)


def find_tray(frame):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for detecting orange
    lower_orange = np.array([5, 100, 100])  # Example lower boundary
    upper_orange = np.array([15, 255, 255])  # Example upper boundary

    # Create a mask for the orange color
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        area = w * h  # Calculate the area of the tray
        return (x, y, w, h, area)
    else:
        return None


def find_coins(frame):
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Save grayscale image
    cv2.imwrite('gray.png', gray)

    # Apply Gaussian blur with a smaller kernel and standard deviation to reduce noise and smooth the image
    blurred = cv2.GaussianBlur(gray, (9, 9), 1)

    # Save blurred image
    cv2.imwrite('blurred.png', blurred)

    # Use adaptive thresholding to create a binary image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Save thresholded image
    cv2.imwrite('thresh.png', thresh)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_circles = []
    coin_areas = []

    # Loop over the contours
    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # The contour is considered a coin if it is sufficiently circular
        if len(approx) > 5:
            # Compute the bounding box of the contour and use it to compute the aspect ratio
            (x, y), radius = cv2.minEnclosingCircle(contour)
            area = cv2.contourArea(contour)
            circularity = 4 * math.pi * (area / (perimeter * perimeter))

            # Check if the circularity is within an acceptable range and the area is not too small
            if 0.7 < circularity < 1.2 and area > 100:  # Thresholds may need tuning
                # Further filter out non-coins by checking the mean intensity of the edges
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_val = cv2.mean(gray, mask=mask)[0]

                if mean_val < 200:  # Threshold may need tuning based on coin brightness
                    found_circles.append((int(x), int(y), int(radius)))
                    coin_areas.append(area)

                    # Save mask of each contour
                    cv2.imwrite(f'mask_{int(x)}_{int(y)}.png', mask)

    # Concatenate all saved images into one
    image_paths = ['gray.png', 'blurred.png', 'thresh.png'] + \
                  [f'mask_{int(c[0])}_{int(c[1])}.png' for c in found_circles]
    images = [Image.open(x) for x in image_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    concatenated_image_path = 'concatenated_image.png'
    new_im.save(concatenated_image_path)

    return found_circles, coin_areas, concatenated_image_path


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
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


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
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    # Convert the HSV colorspace
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only blue color
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    cv2.imshow('obrazek', mask)


def hsv_bitwais():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 0, 0])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('obrazek', res)


def hsv_median():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 0, 0])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    res = cv2.bitwise_and(image, image, mask=mask)
    res = cv2.medianBlur(res, ksize=ksize)
    cv2.imshow('obrazek', res)


def morphology():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    kernel = np.ones((ksize, ksize), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('obrazek', mask_without_noise)


def morphology2():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    kernel = np.ones((7, 7), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    kernel = np.ones((ksize, ksize), np.uint8)
    mask_closed = cv2.morphologyEx(mask_without_noise, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('obrazek', mask_closed)


def marker():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')

    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 0, 0])
    upper = np.array([high_color, 255, 255])

    mask = cv2.inRange(hsv_frame, lower, upper)
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    M = cv2.moments(contours[0])
    print(M)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    image_marker = image.copy()
    cv2.drawMarker(image_marker, (int(cx), int(cy)), color=(
        0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
    cv2.imshow('obrazek', image_marker)


def change_h(x):
    global fun
    if fun is not None:
        fun()

image = None
fun = None
files = None


def main():
    files = os.listdir('pliki')
    image = cv2.imread('pliki/{}'.format(files[0]))  # Make sure to handle file not found errors
    tray = find_tray(image)

    if tray is not None:
        x, y, w, h, area = tray
        # Calculate the center of the rectangle
        center = (x + w // 2, y + h // 2)
        draw_cross(image, center)
        # Optionally, draw the rectangle as well
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Find coins and their areas
    found_circles, areas = find_coins(image)

    # Draw the circles and areas on the image
    for (x, y, r), area in zip(found_circles, areas):
        # Draw the circle in the output image
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        # Draw a rectangle to show the center of the circle
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # Optionally, print the area of the coin
        print(f"Coin at ({x}, {y}) with radius {r}: Area = {area}")

    # Show the image
    cv2.imshow('Coins and Tray Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
