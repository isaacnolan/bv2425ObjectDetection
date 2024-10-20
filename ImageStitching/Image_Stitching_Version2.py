import cv2
import numpy as np
from imutils import paths

images_path = "images/mountain"
output_path = "output/stitchedNew.jpg"

#crop option to create perfect borders
crop = True

#load images
print("[INFO] loading images...")
image_paths = sorted(list(paths.list_images(images_path)))
images = [cv2.imread(image_path) for image_path in image_paths]


#resize images if they are too large
widthThreshold = 1500
resized_images = []
for image in images:
    if image.shape[1] > widthThreshold:
        ratio = widthThreshold / image.shape[1]
        dim = (widthThreshold, int(image.shape[0] * ratio))
        resized_images.append(cv2.resize(image, dim))
    else:
        resized_images.append(image)


#preprocess images to improve clarity and make key deteciton better
print("[INFO] preprocessing images to improve clarity...")
for i in range(len(resized_images)):
    #apply CLAHE to equalize brightness and make sharper
    lab = cv2.cvtColor(resized_images[i], cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    resized_images[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

#stitch images together
#use SCANS which is optimal for ariel footage
print("[INFO] stitching images...")
stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
(status, stitched) = stitcher.stitch(resized_images)

#check status of stitching
if status == cv2.Stitcher_OK:
    print("[INFO] image stitching successful!")
    #cropping
    if crop:
        print("[INFO] cropping...")
        #add a 10px border around the image
        stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
        #convert to grayscale and threshold
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #find contours and crop
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            stitched = stitched[y:y + h, x:x + w]
    #save and display the stitched image
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
