import cv2
import numpy as np


def read_video_frames(video_path, start_frame, end_frame, every_n_frames):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame > total_frames:
        end_frame = total_frames

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (current_frame - start_frame) % every_n_frames == 0:
            frames.append(frame)
        
        current_frame += 1
    
    cap.release()
    return np.array(frames)


def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))
    bounding_boxes = []
    for (x, y, w, h) in faces:
        bounding_boxes.append((x, y, x + w, y + h))
    return bounding_boxes


def draw_boxes(image, boxes, labels=None):
    for i, box in enumerate(boxes):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
        if labels and i < len(labels):
            label = labels[i]
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top_left = (box[0], box[1] - label_size[1] - 10)
            bottom_right = (box[0] + label_size[0], box[1])
            cv2.rectangle(image, top_left, bottom_right, color=(0, 0, 255), thickness=cv2.FILLED)
            cv2.putText(image, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 255, 255), thickness=1)


def save_raw_crops():
    frames = read_video_frames('ap.mp4', 0, 60000, 40)
    crop_id = 0
    for frame in frames:
        for x1, y1, x2, y2 in detect_faces(frame):
            crop = frame[y1:y2, x1:x2]
            cv2.imwrite(f'crops/crop_{crop_id}.png', crop)
            crop_id += 1

def test_detector_on_single_frame():
    # a = 25000
    # a = 35000
    a = 55000
    frames = read_video_frames('ap.mp4', a, a+1, 1)
    frame = frames[0]
    boxes = detect_faces(frame)
    draw_boxes(frame, boxes, ['TODO -- label' for _ in range(len(boxes))])
    cv2.imwrite('vis.png', frame)


if __name__ == '__main__':
    # test_detector_on_single_frame()
    save_raw_crops()
