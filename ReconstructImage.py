 # Reconstruct the original image from the Laplacian pyramid
reconstructed_img = laplacian_pyramid[0]
for i in range(1, len(laplacian_pyramid)):
    expanded = cv2.pyrUp(reconstructed_img)
    # Ensure sizes match before addition
    if expanded.shape != laplacian_pyramid[i].shape:
        expanded = cv2.resize(expanded, (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]))
    reconstructed_img = cv2.add(expanded, laplacian_pyramid[i])

# Display the reconstructed image
cv2_imshow(reconstructed_img)
