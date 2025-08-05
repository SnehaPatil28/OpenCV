import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load an image (replace with your image path)
image_path = '/content/drive/MyDrive/Colab Notebooks/img/41R87izpOGL._UF350,350_QL80_.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Image not found at {image_path}")
else:
    # Get image dimensions
    height, width = img.shape[:2]

    # Define the translation amounts (in pixels)
    tx = 50  # Shift 50 pixels to the right
    ty = 30  # Shift 30 pixels down

    # Create the translation matrix
    # The matrix is of the form:
    # [[1, 0, tx],
    #  [0, 1, ty]]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

    # Apply the translation
    translated_img = cv2.warpAffine(img, translation_matrix, (width, height))

    # Display the original and translated images
    print("Original Image:")
    cv2_imshow(img)
    print("Translated Image:")
    cv2_imshow(translated_img)
