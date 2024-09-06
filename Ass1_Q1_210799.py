import cv2
import numpy as np

def returncontours(edged):
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    return contours

def rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def solution(image_path):
    img = cv2.imread('/content/1.png')
    img_original = img.copy()
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, value = 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 21, 37, 40)
    edged = cv2.Canny(gray, 10, 20)

    contours = returncontours(edged)

    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    points = biggest.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")

    points_sum = points.sum(axis=1)
    input_points[0] = points[np.argmin(points_sum)]
    input_points[3] = points[np.argmax(points_sum)]

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]
    input_points[2] = points[np.argmax(points_diff)]

    converted_points = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
    cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)
    img_original = rgb_to_bgr(img_original)
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)

    img_output = cv2.warpPerspective(img_original, matrix, (600, 600))
    
    img_output = bgr_to_rgb(img_output)

    return img_output