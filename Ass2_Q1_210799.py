import cv2
import numpy as np

def solution(image_path):
    image= cv2.imread(image_path)

    contour_mask = image
    if image is not None:
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for lava color in HSV
        lower_red = np.array([0,  125, 125])
        upper_red = np.array([50, 255, 255])

        # Create a mask using the defined color range
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Perform erosion and dilation to refine the segmentation
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=2)
        dilated = cv2.dilate(eroded, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty mask to draw and fill the contours
        contour_mask = np.zeros_like(image)

        # Draw and fill the contours on the mask
        cv2.drawContours(contour_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

        # Perform morphology to make the contours thinner and clearer
        kernel = np.ones((3, 3), np.uint8)
        contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    return contour_mask
