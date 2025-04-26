import cv2
import numpy as np
from ultralytics import YOLO
model = YOLO('yolo12m.pt')  # Use 'yolov8n.pt' for detection-only, 'yolov8n-seg.pt' for segmentation

def detect_objects(tile):
    """Run YOLO detection on a tile and return tile with drawn boxes and labels."""
    results = model.predict(tile, verbose=False)[0]
    annotated_tile = tile.copy()

    if results.masks is not None:
        # If segmentation masks are available
        masks = results.masks.xy
        for mask in masks:
            points = np.array(mask, dtype=np.int32)
            cv2.polylines(annotated_tile, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(annotated_tile, [points], color=(0, 255, 0, 50))  # translucent mask

    if results.boxes is not None:
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            confidence = float(conf)

            # Get label name
            label = model.names[class_id] if hasattr(model, "names") else str(class_id)
            label_text = f"{label} {confidence:.2f}"

            # Draw rectangle
            cv2.rectangle(annotated_tile, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Put label above the box
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_tile, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (255, 0, 0), -1)
            cv2.putText(annotated_tile, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated_tile

  

def process_video(input_path, output_path=None):
    # Open video file
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional: Prepare video writer if you want to save output
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on the frame
        output_frame = detect_objects(frame)

        # Display the frame (optional)
        cv2.imshow('Detection', output_frame)

        # Write frame to output if needed
        if out:
            out.write(output_frame)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = "DJI_0029.mp4"      # Change this to your video path
    output_video = "output.mp4"     # Change or set to None if you don't want to save
    process_video(input_video, output_video)
