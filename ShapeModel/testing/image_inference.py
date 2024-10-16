import cv2
import time
from ultralytics import YOLO

def main():
    # Load the YOLOv11 model
    model = YOLO('yolov11m.pt')

    # Load the image from file
    image_path = '~/basketball_test.jpg'
    frame = cv2.imread(image_path)

    if frame is None:
        print("Error: Could not load image.")
        return

    start_time = time.time()
    # Run inference on the single image
    results = model(frame)

    end_time = time.time()

    inference_time = end_time - start_time
    print(f'Inference Time: {inference_time:.4f} seconds')
    # Plot results (draw bounding boxes and labels on the image)
    annotated_frame = results[0].plot()

    # Display the image with annotations
    cv2.imshow('YOLOv11 Image Inference', annotated_frame)

    # Wait until a key is pressed to close the window
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
