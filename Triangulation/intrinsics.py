import cv2
import numpy as np
import glob

# Define the chessboard size (inner corners)
CHESSBOARD_SIZE = (7, 6)

# Prepare 3D points for a single chessboard image
# like (0,0,0), (1,0,0), (2,0,0) ... assuming 1 unit = 1 square size
objp = []
for i in range(CHESSBOARD_SIZE[1]):
    for j in range(CHESSBOARD_SIZE[0]):
        objp.append([j, i, 0])
objp = np.array(objp, dtype=np.float32)

# Arrays to store object points and image points from all images
obj_points = []  # 3D points
img_points = []  # 2D points

# Load all images in a folder (just an example pattern)
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        # Refine corner locations
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        # Add points
        obj_points.append(objp)
        img_points.append(corners_refined)

        # Optional: draw and display the corners
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners_refined, found)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Calibrate camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

print("Camera Matrix:\n", camera_matrix)
print("Distortion Coeffs:\n", dist_coeffs)
print("Reprojection Error:\n", ret)
