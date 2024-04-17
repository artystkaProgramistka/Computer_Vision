import cv2

import numpy as np

MAX_COIN_DIAMETER_PIXELS = 100


def get_filtered_mask_for_edges(edge):
    dilated = 255 - cv2.dilate(edge, np.ones((3, 3)), iterations=1)
    n_comp, components_mask = cv2.connectedComponents(dilated)
    filtered_mask = np.zeros_like(edge)
    for i in range(1, n_comp):
        mask = (components_mask == i).astype(np.uint8) * 255
        x1, y1, x2, y2 = cv2.boundingRect(mask)
        if x2 - x1 <= MAX_COIN_DIAMETER_PIXELS and y2 - y1 <= MAX_COIN_DIAMETER_PIXELS:
            filtered_mask[mask > 0] = 255
    return filtered_mask


def find_tray(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)
    return None


def process_one_img(img_path):
    color_img = cv2.imread(img_path)
    tray = find_tray(color_img)
    if tray is None:
        print('Tray not found!')
        return None

    tx, ty, tw, th = tray
    cv2.rectangle(color_img, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)

    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    edge1 = cv2.Canny(gray_img, 40, 80)
    edge2 = cv2.Canny(gray_img, 80, 120)
    edge3 = cv2.Canny(gray_img, 100, 140)
    edge4 = cv2.Canny(gray_img, 140, 200)

    filtered_mask1 = get_filtered_mask_for_edges(edge1)
    filtered_mask2 = get_filtered_mask_for_edges(edge2)
    filtered_mask3 = get_filtered_mask_for_edges(edge3)
    filtered_mask4 = get_filtered_mask_for_edges(edge4)
    filtered_mask = np.max([filtered_mask1, filtered_mask2, filtered_mask3, filtered_mask4], axis=0)

    processed_mask = cv2.dilate(filtered_mask, np.ones((3, 3)), iterations=7)
    processed_mask = cv2.erode(processed_mask, np.ones((3, 3)), iterations=21)

    visualization = np.concatenate([
        cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR),  # Convert gray_img to color for consistent visualization
        cv2.cvtColor(edge1, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR),
        color_img  # Add the color image with the rectangle at the end of the visualization stack
    ], axis=0)

    return processed_mask, visualization, tray # Return the color image with the tray highlighted


def get_coin_areas(processed_mask):
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coin_areas = [cv2.contourArea(contour) for contour in contours]
    return coin_areas

def count_coins(processed_mask, tray):
    tx, ty, tw, th = tray
    tray_area = tw * th
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coin_areas = [cv2.contourArea(contour) for contour in contours]

    threshold_percentage = 0.0013  # Coins with area > 5% of tray area are "large"
    large_coin_count = 0
    small_coin_count = 0

    for area in coin_areas:
        if area > tray_area * threshold_percentage:
            large_coin_count += 1
        else:
            small_coin_count += 1

    on_tray_count = 0
    off_tray_count = 0
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if tx <= cx <= tx + tw and ty <= cy <= ty + th:
                on_tray_count += 1
            else:
                off_tray_count += 1
    return large_coin_count, small_coin_count, on_tray_count, off_tray_count


def main():
    coins_visualizations = []
    for i in range(1, 9):
        processing_output = process_one_img(f'./pliki/tray{i}.jpg')
        if processing_output is not None:
            coins_detection_mask, visualization, tray = processing_output
            coins_visualizations.append(visualization)

            large_coin_count, small_coin_count, on_tray_count, off_tray_count = count_coins(coins_detection_mask, tray)

            # Save the visualization of all trays
            cv2.imwrite('output/visualization.png', np.concatenate(coins_visualizations, axis=1))

            print(f"Large coins on image {i}: {large_coin_count}, Small coins on image {i}: {small_coin_count}")
            print(f"Coins on the tray on image {i}: {on_tray_count}, Cons off the tray on image {i}: {off_tray_count}")

if __name__ == "__main__":
    main()