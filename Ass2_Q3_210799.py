import cv2
import numpy as np


def solution(audio_path):

    image = cv2.imread(audio_path)

    h, w, _ = image.shape
    left_end = None

    for y in range(h - 1, -1, -1):
        for x in range(w):
            if not np.all(image[y, x] == [255, 255, 255]):
                left_end = (x, y)
                break
        if left_end is not None:
            break


    h, w, _ = image.shape
    right_end = None

    for y in range(h - 1, -1, -1):
        for x in range(w-1, -1, -1):
            if not np.all(image[y, x] == [255, 255, 255]):
                right_end = (x, y)
                break
        if right_end is not None:
            break

    center = np.mean([left_end, right_end], axis=0)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    bw_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2GRAY)
    _, blacki_mask = cv2.threshold(hsv_image, 1, 255, cv2.THRESH_BINARY)
    
    x, y = center
    x, y = int(x), int(y)

    roi_width = 50

    lower_skin = np.array([10, 20, 40], dtype=np.uint8)
    upper_skin = np.array([30, 200, 200], dtype=np.uint8)

    left_roi = hsv_image[:, :x-roi_width, :]
    right_roi = hsv_image[:, x+roi_width:, :]

    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([0, 0, 0], dtype=np.uint8)
    
    left_mask_black = cv2.inRange(left_roi, lower_black, upper_black)
    right_mask_black = cv2.inRange(right_roi, lower_black, upper_black)

    left_black_pixels = cv2.countNonZero(left_mask_black)
    right_black_pixels = cv2.countNonZero(right_mask_black)

    black_ratio = (right_black_pixels + 1e-8)/ (left_black_pixels + 1e-8) 
    
    left_mask = cv2.inRange(left_roi, lower_skin, upper_skin)
    right_mask = cv2.inRange(right_roi, lower_skin, upper_skin)
    

    left_skin_pixels = cv2.countNonZero(left_mask)
    right_skin_pixels = cv2.countNonZero(right_mask)
    
    ratio_skin = right_skin_pixels / (left_skin_pixels + 1e-8)

    for angle in range(0, 360, 10):
        hsv_image = cv2.rotate(hsv_image, angle)
    
    class_name = 'fake'
    
    if(ratio_skin < 1.15 and ratio_skin > 1.00 and black_ratio > 0.98 and black_ratio < 1.21):
        class_name = 'real'
    else:
        class_name = 'fake'
    return class_name
