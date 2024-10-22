from ultralytics import YOLO
import cv2
import time
import psutil

#Functions

# Gets ram usage
def get_ram_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB

# Validate the model
def validation(model):
    metrics = model.val(data = "",
                        imgsz = 640,
                        batch = 16,
                        save_hybrid = True,
                        conf = .01,
                        device = "cpu",
                        plots = True,
                        save_json=True)  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category



def main():
    # Initialize models
    models_directory = 'ShapeModel/testing/test_pipelines/base_test_pipeline/models'
    modelNames = ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']

    # Load the image from file
    dataset_dir = 'ShapeModel/testing/data'


    frame = cv2.imread(image_path)

    if frame is None:
        print("Error: Could not load image.")
        return

    results_time = []
    results_memory = []
    # Loop through each model
    for model_name in modelNames:
        print(f"Testing model: {model_name}")
        model_path = models_directory + model_name

        # Load model
        model = YOLO(model_path)

        initial_memory = get_ram_usage()
        start_time = time.perf_counter()

        results = model(frame)

        end_time = time.perf_counter()
        after_memory = get_ram_usage()

        inference_time = end_time - start_time
        memory_usage = after_memory - initial_memory

        # Store results in log
        results_time.append({'model': model_name, 'inference_time': inference_time})
        results_memory.append({'model': model_name, 'memory_usage': memory_usage})

        # Optionally, display or save the annotated image
        annotated_frame = results[0].plot()
        
        # Create a resizable window
        cv2.namedWindow(model_name, cv2.WINDOW_NORMAL)

        # Display the image with annotations
        cv2.imshow(model_name, annotated_frame)

        # Resize the window to match the image dimensions
        cv2.resizeWindow(model_name, frame.shape[1], frame.shape[0])

        # Close window after keypress
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Print out the time data
    for result in      results_time:
        print(f"Model: {result['model']}, Total inference elapsed Time: {result['inference_time']:.4f} seconds")

    # Print out the memory usage
    for usage in results_memory:
        print(f"Model: {usage['model']}, Total change in memory usage: {usage['memory_usage']:.4f} MB")

if __name__ == "__main__":
    main()