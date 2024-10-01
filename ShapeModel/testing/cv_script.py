import cv2
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame) 

        annotated_frame = results[0].plot() 

        cv2.imshow('YOLOv8 Webcam Inference', annotated_frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

