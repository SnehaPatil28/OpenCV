#Perform camera calibration to find intrinsic parameters
import cv2
import numpy as np
import glob

# Prepare object points (0,0,0), (1,0,0), ..., (6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = []  # 3D points
imgpoints = []  # 2D points

# Load all images in the directory
images = glob.glob('/content/drive/MyDrive/Colab Notebooks/img/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # cv2.drawChessboardCorners(img, (7,6), corners, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(100)

# cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
