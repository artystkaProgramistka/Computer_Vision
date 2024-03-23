import cv2
import numpy as np


def draw_cross(img, center, size=20, color=(0, 255, 0), thickness=2):
    # Draw a cross at the specified center point
    cv2.line(img, (center[0] - size, center[1]), (center[0] + size, center[1]), color, thickness)
    cv2.line(img, (center[0], center[1] - size), (center[0], center[1] + size), color, thickness)


def process_frame(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Use morphological operations to improve roundness
    # Apply a circular closing operation to fill holes and maintain roundness
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # Adjust size as needed
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    red_mask = cv2.medianBlur(red_mask, ksize=49)  # Apply median blur

    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If at least one contour is found, find the largest contour assumed to be the ball
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:  # To avoid division by zero
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            draw_cross(frame, (cX, cY))  # Draw the cross on the original frame

    return frame, red_mask

# The rest of the video capture and processing code remains unchanged


# Video capture and processing
video = cv2.VideoCapture('moving_arrow.mov')
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)

result = cv2.VideoWriter('result_arrow.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, size)
mask_result = cv2.VideoWriter('mask_result_arrow.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, size, False)

counter = 1

while True:
    success, frame = video.read()
    if not success:
        break

    processed_frame, ball_mask = process_frame(frame)
    result.write(processed_frame)
    mask_result.write(ball_mask)  # Save the mask video

    counter += 1
    print(f'Frame {counter}/{total_frames}')

video.release()
result.release()
mask_result.release()
