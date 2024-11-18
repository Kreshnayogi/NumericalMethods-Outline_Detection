import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_gray(rgb_image):

    # Get image dimensions
    height, width, _ = rgb_image.shape

    # Initialize grayscale image array
    grayscale_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each pixel
    for y in range(height):
        for x in range(width):
            # Extract RGB values
            red = rgb_image[y, x, 0]
            green = rgb_image[y, x, 1]
            blue = rgb_image[y, x, 2]

            # Apply the NTSC formula for grayscale conversion
            grayscale_value = int(0.2989 * red + 0.5870 * green + 0.1140 * blue)

            # Assign grayscale value to the corresponding pixel
            grayscale_image[y, x] = grayscale_value

    return grayscale_image

def edge_detection(gray_image):
    threshold = 20

    # Get image dimensions
    height, width = gray_image.shape

    # Initialize image edges array
    edges = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each pixel
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Compute second-order derivatives using Taylor series expansion
            dy = (int(gray_image[y + 1, x]) - int(gray_image[y - 1, x]))/2
            dx = (int(gray_image[y, x + 1]) - int(gray_image[y, x - 1]))/2

            # magnitude
            magnitude = np.sqrt(dx*dx + dy*dy)

            # Apply threshold
            if magnitude > threshold:
                edges[y, x] = 255
            else:
                edges[y, x] = 0


    return edges

# Path to the image file on Google Drive
rgb_image_path = 'NumericalMethods-Outline_Detection-main/cow.jpg'

# Read the RGB image
rgb_image = cv2.imread(rgb_image_path)

gray_image = rgb_to_gray(rgb_image)

# Apply edge detection using Taylor series approximation
edges = edge_detection(gray_image)

# Display the edging
plt.imshow(gray_image, cmap='gray')
plt.axis('off')  # Hide axes
plt.show()

# Display the edging
plt.imshow(edges, cmap='gray')
plt.axis('off')  # Hide axes
plt.show()
