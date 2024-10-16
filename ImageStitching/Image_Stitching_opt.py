import cv2
from imutils import paths

# Set the path to the directory containing images
images_path = "images/mountain"

# Set the output image path with a valid image extension
output_path = "output/stitched.jpg"

# Set crop option: True to enable cropping, False to disable
crop = True  # Change to True if you want to crop the stitched image

# Load images
print("[INFO] loading images...")
image_paths = sorted(list(paths.list_images(images_path)))
images = [cv2.imread(image_path) for image_path in image_paths]

# Stitch images
print("[INFO] stitching images...")
stitcher = cv2.Stitcher_create()
status, stitched = stitcher.stitch(images)

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
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        # Find contours and crop
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            stitched = stitched[y:y + h, x:x + w]
    # Save and display the stitched image
    cv2.imwrite(output_path, stitched)
    cv2.imshow("Stitched Image", stitched)
    cv2.waitKey(0)
else:
    print(f"[INFO] image stitching failed ({status})")
