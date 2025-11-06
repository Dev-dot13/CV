import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# 1. Define chessboard dimensions (inner corners)
chessboard_size = (9, 6)  # 9x6 inner corners in the calibration pattern
square_size = 1.0         # Set this to your real square size (e.g., 1 cm)

# 2. Prepare object points like (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all calibration images
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# 3. Load calibration images (taken from different angles)
images = glob.glob('calib_images/*.jpg')  # Folder containing chessboard images

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Corners in {fname}")
        plt.axis('off')
        plt.show()