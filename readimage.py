from google.colab import drive
drive.mount('/content/drive')

import cv2 as cv
import sys
from google.colab.patches import cv2_imshow
import os # Import the os module

# Attempting to load image directly with the path
image_path = "/content/drive/MyDrive/Colab Notebooks/Image1.jpg"

# Check if the file exists
if not os.path.exists(image_path):
    sys.exit(f"Error: Image file not found at {image_path}")

img = cv.imread(image_path)

if img is None:
    sys.exit("Could not read the image.")

res = cv.resize(img,None,fx=0.1, fy=0.1, interpolation = cv.INTER_CUBIC)

cv2_imshow(img)
