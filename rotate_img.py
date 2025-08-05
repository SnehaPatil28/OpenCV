import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load an image (you can replace this with your image path)
# For demonstration, let's assume you have an image named 'cr.jpg' in your drive as used previously
image_path = '/content/drive/MyDrive/Colab Notebooks/img/41R87izpOGL._UF350,350_QL80_.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Image not found at {image_path}")
else:
    # Get image dimensions
    height, width = img.shape[:2]

    # Define the rotation angle (in degrees)
    angle = 45

    # Get the rotation matrix
    # The arguments are: center of rotation, angle, and scale
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

    # Apply the rotation
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))

    # Display the original and rotated images
    print("Original Image:")
    cv2_imshow(img)
    print("Rotated Image:")
    cv2_imshow(rotated_img)
