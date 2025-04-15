from collections import defaultdict
import torch
from ultralytics import YOLO

def filter_objects(images, model_path, min_occurrences=3):
    """
    Filters out objects detected in less than `min_occurrences` frames.
    
    :param images: List of image file paths or numpy arrays.
    :param model_path: Path to the YOLO model.
    :param min_occurrences: Minimum number of times an object must appear to be retained.
    :return: Dictionary of object labels with occurrences >= min_occurrences.
    """
    model = YOLO(model_path)  # Load YOLO model
    object_counts = defaultdict(int)
    
    for image in images:
        results = model(image)  # Run YOLO model
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls)]  # Get label name
                object_counts[label] += 1
    
    # Filter out objects with fewer than min_occurrences
    filtered_objects = {label: count for label, count in object_counts.items() if count >= min_occurrences}
    
    return filtered_objects

# Example usage
# images = ["image1.jpg", "image2.jpg", "image3.jpg"]
# model_path = "yolov8.pt"  # Replace with your model path
# filtered_results = filter_objects(images, model_path)
# print(filtered_results)
