import cv2

def match_images(image1, image2, algorithm='SIFT'):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    if algorithm == 'SIFT':
        detector = cv2.SIFT_create()
    elif algorithm == 'ORB':
        detector = cv2.ORB_create()
    else:
        raise ValueError("Unsupported algorithm: choose 'SIFT' or 'ORB'")

    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

    if algorithm == 'SIFT':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif algorithm == 'ORB':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_images = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matching_score = len(matches) / (len(keypoints1) + len(keypoints2))

    return matching_score, matched_images

def process_frame(frame, images, algorithm='SIFT'):
    highest_score = 0
    result_matched_images = None
    for img in images:
        score, matched_images = match_images(frame, img, algorithm)
        if score > highest_score:
            highest_score = score
            result_matched_images = matched_images
    return result_matched_images

def resize_images_to_match_video(images, video_size):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, video_size)
        resized_images.append(resized_img)
    return resized_images

def check_image_sizes(video):
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_size = (frame_width, frame_height)
    return video_size

def process_video(video_path, images, algorithm, output_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError("Error opening video file")

    video_size = check_image_sizes(video)
    resized_images = resize_images_to_match_video(images, (video_size[0], video_size[1]))

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width * 2, frame_height)

    result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

    counter = 1

    while True:
        success, frame = video.read()
        if not success:
            break

        processed_frame = process_frame(frame, resized_images, algorithm)
        if processed_frame is not None:
            result.write(processed_frame)

        counter += 1
        print(f'Frame {counter}/{total_frames}')

    video.release()
    result.release()

def main():
    images = [cv2.imread(f'pliki/{i}.jpeg') for i in range(1, 10)]
    if any(img is None for img in images):
        raise FileNotFoundError("One or more image files could not be loaded")

    # Process with SIFT
    print("Processing with SIFT...")
    process_video('pliki/video.mp4', images, 'SIFT', 'output/result_sift.mp4')

    # Process with ORB
    print("Processing with ORB...")
    process_video('pliki/video.mp4', images, 'ORB', 'output/result_orb.mp4')

if __name__ == '__main__':
    main()
