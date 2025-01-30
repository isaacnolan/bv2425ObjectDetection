from ultralytics import YOLO
import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("GPU Name:", torch.cuda.get_device_name(0))

model = YOLO('yolov8n.pt')  # Load a small YOLOv8 model
print(model)

