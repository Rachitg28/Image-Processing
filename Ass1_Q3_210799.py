import cv2
import numpy as np

def solution(image_path):
    # Load the image
    image = cv2.imread(image_path)
    orig = image.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    gray = cv2.bitwise_not(gray)

    # Threshold the image to create a binary mask of the text
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find the coordinates of the white pixels (text)
    coords = np.column_stack(np.where(thresh > 0))

    # Get the minimum bounding rectangle (minRect)
    minRect = cv2.minAreaRect(coords)
    box = cv2.boxPoints(minRect)
    box = np.int0(box)

    # Shift the minRect to the right by 10 pixels
    box[:, 0] += 10

    # Draw a green boundary around the shifted minRect
    cv2.drawContours(orig, [box], 0, (0, 255, 0), 2)

    # Detect lines in the image using the Hough Line Transform
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    # Find the longest line
    longest_line = None
    max_length = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if length > max_length:
            longest_line = line
            max_length = length

    # Calculate the angle of inclination of the longest line
    x1, y1, x2, y2 = longest_line[0]
    angle = np.arctan2(y2 - y1, x2 - x1)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle)

    # Rotate the image around the calculated angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Check if the longest line is at the bottom, and if so, rotate by 180 degrees
    if y1 > h // 2:
        angle_degrees += 180

    # Increase the size of the image to enclose all text
    scale_factor = 1.5  # You can adjust this value to control the size increase
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Draw lines on the original image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green lines
    cv2.imshow("rotated", rotated)
    cv2.waitKey(0)

    return rotated