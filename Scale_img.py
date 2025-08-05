import cv2
from google.colab.patches import cv2_imshow

# Load an image (replace with your image path)
image_path = '/content/drive/MyDrive/Colab Notebooks/img/41R87izpOGL._UF350,350_QL80_.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Image not found at {image_path}")
else:
    # Define scaling factors
    scale_factor_x = 0.5  # Scale to 50% of original width
    scale_factor_y = 0.5  # Scale to 50% of original height

    # Scale the image
    scaled_img = cv2.resize(img, None, fx=scale_factor_x, fy=scale_factor_y, interpolation = cv2.INTER_LINEAR)

    # Display the original and scaled images
    print("Original Image:")
    cv2_imshow(img)
    print("Scaled Image:")
    cv2_imshow(scaled_img)
