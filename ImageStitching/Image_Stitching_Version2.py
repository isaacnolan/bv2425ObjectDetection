import cv2
import numpy as np
from imutils import paths

# Set the path to the directory containing images
images_path = "images/mountain"

# Set the output image path with a valid image extension
output_path = "output/stitchedNew.jpg"

# Set crop option: True to enable cropping, False to disable
crop = True

# Load images
print("[INFO] loading images...")
image_paths = sorted(list(paths.list_images(images_path)))
images = [cv2.imread(image_path) for image_path in image_paths]

# Resize images if they are too large to manage memory efficiently
resized_images = []
for image in images:
    if image.shape[1] > 1500:  # Resize images with width greater than 1500 pixels
        ratio = 1500.0 / image.shape[1]
        dim = (1500, int(image.shape[0] * ratio))
        resized_images.append(cv2.resize(image, dim))
    else:
        resized_images.append(image)

# Preprocess images to improve clarity
print("[INFO] preprocessing images to improve clarity...")
for i in range(len(resized_images)):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(resized_images[i], cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    resized_images[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Stitch images using SCANS mode
print("[INFO] stitching images...")
stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
(status, stitched) = stitcher.stitch(resized_images)

# Check if stitching was successful
if status == cv2.Stitcher_OK:
    print("[INFO] image stitching successful!")
    # Optional cropping
    if crop:
        print("[INFO] cropping...")
        # Add a 10px border around the image
        stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Find contours and crop
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            stitched = stitched[y:y + h, x:x + w]
    # Save and display the stitched image
    cv2.imwrite(output_path, stitched)
    print(f"[INFO] stitched image saved to {output_path}")
    cv2.imshow("Stitched Image", stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"[ERROR] image stitching failed (error code: {status})")
    if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("[SOLUTION] Need more images to perform stitching.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("[SOLUTION] Homography estimation failed. Try improving image overlap or reducing blur.")
    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        print("[SOLUTION] Camera parameter adjustment failed. Try changing image capture settings.")
