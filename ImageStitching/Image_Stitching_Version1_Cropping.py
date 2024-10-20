import cv2
from imutils import paths
import os

# Set the path to the directory containing images
images_path = "images/mountain"

# Set the output image path with a valid image extension
output_path = "output/stitched_cropped.jpg"

# Set crop option: True to enable cropping, False to disable
crop = True  # Ensure this is set to True to enable cropping

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load images
print("[INFO] Loading images...")
image_paths = sorted(list(paths.list_images(images_path)))
images = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARNING] Unable to read image: {image_path}. Skipping.")
    else:
        images.append(image)

if not images:
    print("[ERROR] No images to stitch. Exiting.")
    exit()

# Stitch images
print("[INFO] Stitching images...")
stitcher = cv2.Stitcher_create()
status, stitched = stitcher.stitch(images)

# Check if stitching was successful
if status == cv2.Stitcher_OK:
    print("[INFO] Image stitching successful!")
    # Optional cropping
    if crop:
        print("[INFO] Cropping stitched image...")
        # Convert to grayscale
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        # Apply Otsu's thresholding to create a binary mask
        _, thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Select the largest contour based on area
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            print(f"[INFO] Cropping to bounding rectangle - x: {x}, y: {y}, w: {w}, h: {h}")
            # Crop the image to the bounding rectangle
            stitched_cropped = stitched[y:y + h, x:x + w]
        else:
            print("[WARNING] No contours found. Saving the original stitched image without cropping.")
            stitched_cropped = stitched
    else:
        stitched_cropped = stitched
    
    # Save the final cropped stitched image
    cv2.imwrite(output_path, stitched_cropped)
    print(f"[INFO] Stitched image saved to {output_path}")
    
    # Display the final cropped stitched image
    cv2.imshow("Final Cropped Stitched Image", stitched_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"[ERROR] Image stitching failed with status code {status}.")
    # Optional: Decode the status code to get a human-readable message
    if status == cv2.Stitcher_ERR_NEED_MORE_IMAGES:
        print("Need more images to stitch.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("Homography estimation failed.")
    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        print("Camera parameters adjustment failed.")
    else:
        print("Unknown error.")
