import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

CHESSBOARD_SIZE = (7, 7) # number inner corners
SQUARE_SIZE = 24.5  # chessboard square size in mm

# Prepare object points (0,0,0), (1,0,0), ..., (8,5,0) scaled by square size
objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image

# Change this to your chessboard image directory
images = glob.glob('./calibration_images/*.jpg')
if len(images) == 0:
    print("No images found in 'calibration_images' directory. Please add chessboard images for calibration.")
    exit(1)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')
    plt.show()
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(100)
cv2.destroyAllWindows()

if len(objpoints) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    print('Camera matrix:\n', camera_matrix)
    print('Distortion coefficients:\n', dist_coeffs)
    # Save results
    np.savez('camera_calibration.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print('Calibration saved to camera_calibration.npz')
else:
    print('No chessboard corners found. Check your images and chessboard size.')
