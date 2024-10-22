from ultralytics import YOLO

# Load a model
model = YOLO("path/to/best.pt")  # load a custom model

# Validate the model
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