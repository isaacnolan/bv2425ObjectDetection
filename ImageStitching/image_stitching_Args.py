# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
                help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to the output image")
ap.add_argument("-c", "--crop", type=int, default=0,
                help="whether to crop out the largest rectangular region")
args = vars(ap.parse_args())

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

# loop over the image paths, load each one, and add them to our list
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)

# initialize OpenCV's image stitcher object and perform the image stitching
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# if the stitching was successful, crop the image if required
if status == cv2.Stitcher_OK:
    print("[INFO] image stitching successful!")
    
    if args["crop"] > 0:
        print("[INFO] cropping...")
        # Add a 10px border around the image (in case we need to find contours)
        stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        
        # Convert to grayscale and threshold the image to find the contours
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        
        # Find the external contours in the threshold image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        
        # Get the bounding box of the largest contour and use it to crop the image
        (x, y, w, h) = cv2.boundingRect(c)
        stitched = stitched[y:y + h, x:x + w]

    # Save the output image
    cv2.imwrite(args["output"], stitched)
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)

else:
    print(f"[INFO] image stitching failed ({status})")
