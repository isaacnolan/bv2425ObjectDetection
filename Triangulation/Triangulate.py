import cv2
import numpy as np

def main():
    # Example 2D point correspondences in two images.
    # Here, we have 4 matched points across 2 images.
    # Each 2D array is of shape (2, N).
    #given from suryas code
    points2d_img1 = np.array([
        [100, 200, 300, 400],  # x-coordinates
        [100, 120, 100, 120]   # y-coordinates
    ], dtype=np.float32)

    points2d_img2 = np.array([
        [110, 210, 310, 410],   # x-coordinates
        [95,  115, 90,  110]    # y-coordinates
    ], dtype=np.float32)

    # Combine into a list of 2D arrays (one per image)
    points2d = [points2d_img1, points2d_img2]

    # Example projection matrices (3x4) for the two images.
    # In a real scenario, these come from camera intrinsic and extrinsic parameters.
    
    K = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    #3×4 projection matrix
    #(K[R∣t])
    #R is all 1 
    
    #fill in once checkerbored image is given
    #fourth column should be coordinates of the drone so the final output is also long and lat
    P1 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float32)

    # Suppose the second camera is translated 100 units along the X-axis.
    P2 = np.array([
        [1.0, 0.0, 0.0, 100.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float32)

    # Combine into a list of projection matrices
    projection_matrices = [P1, P2]

    # Perform triangulation
    points3d_hom = cv2.sfm.triangulatePoints(points2d, projection_matrices)

    
    #Shouldnt need below if statement remove it after testing
    
    # Check if points3d_hom has shape (4, N)
    if points3d_hom.shape[0] == 4:
        # Convert from homogeneous to Euclidean: X = X_h / W
        points3d = points3d_hom[:3] / points3d_hom[3]
    else:
        # If it's 3 x N, they are already in Euclidean coords
        points3d = points3d_hom
    
    print("Triangulated 3D points (shape = {}):\n".format(points3d.shape))
    print(points3d)

if __name__ == "__main__":
    main()
