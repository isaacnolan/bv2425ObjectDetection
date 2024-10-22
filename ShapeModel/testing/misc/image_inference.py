# Adarsha's script
import cv2
import time
from ultralytics import YOLO

def main():
    # Load the YOLO model
    modelsDirectory = 'C:\\Users\\adars\\repos\\bv2425ObjectDetection\\ShapeModel\\testing\\misc\\'
    modelName = 'yolov8n'
    model = YOLO(modelsDirectory + modelName)

    # Load the image from file
    image_path = 'C:\\Users\\adars\\repos\\bv2425ObjectDetection\\ShapeModel\\testing\\data\\basketball_court.jpg'

    frame = cv2.imread(image_path)

    if frame is None:
        print("Error: Could not load image.")
        return

    # Run inference and calc inference time
    start_time = time.perf_counter()
    results = model(frame)
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    print(f'Inference Time: {inference_time:.4f} seconds')

    # Plot results (draw bounding boxes and labels on the image)
    annotated_frame = results[0].plot()

    # Display the image with annotations
    cv2.imshow(modelName, annotated_frame)

    # Wait until a key is pressed to close the window
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
