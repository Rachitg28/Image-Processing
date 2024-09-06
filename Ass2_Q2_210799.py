import cv2
import numpy as np

def gaussian(x, sigma):
    return np.exp(-x**2 / (2 * sigma**2))

def bilateral_filter(image, d, sigma_color, sigma_space):
    height, width, _ = image.shape
    result = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            i_min, i_max = max(0, i - d), min(height, i + d + 1)
            j_min, j_max = max(0, j - d), min(width, j + d + 1)

            region = image[i_min:i_max, j_min:j_max]

            intensity_diff = np.linalg.norm(region - image[i, j], axis=2)
            spatial_diff_i, spatial_diff_j = np.meshgrid(
                np.arange(i_min, i_max) - i,
                np.arange(j_min, j_max) - j,
                indexing='ij'
            )
            spatial_diff = np.sqrt(spatial_diff_i**2 + spatial_diff_j**2)

            weight = (
                gaussian(intensity_diff, sigma_color) *
                gaussian(spatial_diff, sigma_space)
            )

            normalized_weight = weight / np.sum(weight)

            result[i, j] = np.sum(region * normalized_weight[:, :, None], axis=(0, 1))

    return result

def combine_images(image1, image2, weight):
    return weight * image1 + (1 - weight) * image2

def solution(image_path_a, image_path_b):
    flash_image = cv2.imread(image_path_a)
    nonflash_image = cv2.imread(image_path_b)

    # Set bilateral filter parameters
    d = 10
    sigma_color = 15
    sigma_space = 15

    # Apply bilateral filter to both images
    flash_filtered = bilateral_filter(flash_image, d, sigma_color, sigma_space)
    nonflash_filtered = bilateral_filter(nonflash_image, d, sigma_color, sigma_space)

    # Calculate the pixel-wise weighted sum
    alpha = 0.5
    combined_image = combine_images(flash_filtered, nonflash_filtered, alpha)

    return combined_image.astype(np.uint8)
