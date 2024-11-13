import cv2
import numpy as np
from imutils import paths
#images should be taken linearly and not panoramically
#image file names must be named accordingly in increasing order (first photo as the lowest)

#possible changes: change the feature detection from whatever openCV uses to Sift or ORB etc.

#directory that has images in correct order
images_path = "ImageStitching/Python/images/googleMaps"
output_path = "ImageStitching/Python/output/stitchedNew.jpg"

#crop option to create perfect borders
crop = True
#option to increase visibilty of key points with image preprocessing
preprocessing = False
def main():
    images = load_images(images_path)
    resized_images = resize_images(images)
    resized_images = preprocess_images(resized_images)
    status, stitched = stitch_images(resized_images)

    #check status of stitching
    if status == cv2.Stitcher_OK:
        print("[INFO] image stitching successful!")
        stitched = crop_image(stitched)
        save_and_display_image(stitched, output_path)
    else:
        print(f"[ERROR] image stitching failed (error code: {status})")
        if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            print("[SOLUTION] Need more images to perform stitching.")
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            print("[SOLUTION] Homography estimation failed. Try improving image overlap or reducing blur.")
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            print("[SOLUTION] Camera parameter adjustment failed. Try changing image capture settings.")

def load_images(images_path):
    #load images
    print("[INFO] loading images...")
    image_paths = sorted(list(paths.list_images(images_path)))
    images = [cv2.imread(image_path) for image_path in image_paths]
    return images

def resize_images(images, widthThreshold=1500):
    #resize images if they are too large
    resized_images = []
    for image in images:
        if image.shape[1] > widthThreshold:
            print("[INFO] Resizing Image...")
            ratio = widthThreshold / image.shape[1]
            dim = (widthThreshold, int(image.shape[0] * ratio))
            resized_images.append(cv2.resize(image, dim))
        else:
            resized_images.append(image)
    return resized_images

def preprocess_images(resized_images):
    #preprocess images to improve clarity and make key deteciton better
    if preprocessing:
        print("[INFO] preprocessing images to improve clarity...")
        for i in range(len(resized_images)):
            #apply CLAHE to equalize brightness and make sharper
            lab = cv2.cvtColor(resized_images[i], cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            resized_images[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return resized_images

def stitch_images(resized_images):
    #stitch images together
    #use SCANS which is optimal for ariel footage
    print("[INFO] stitching images...")
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    (status, stitched) = stitcher.stitch(resized_images)
    return status, stitched

def crop_image(stitched):
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
    return stitched

def save_and_display_image(stitched, output_path):
    #save and display the stitched image
    cv2.imwrite(output_path, stitched)
    print(f"[INFO] stitched image saved to {output_path}")
    cv2.imshow("Stitched Image", stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
