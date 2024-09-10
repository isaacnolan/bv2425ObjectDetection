from IPython.display import display, Image
from ultralytics import YOLO

#load the model
model = YOLO("yolov8m.pt")#loads nano detection YOLO version 8 neural network

n_epochs = 250
bs = -1
gpu_id = 0
imgSize = 640
waitNum = 15
workerNum = 8
OptimizerChoice = 'auto'
validate = True

#Train
results = model.train(data = r"/content/drive/MyDrive/data.yaml",
                      imgsz = imgSize,
                      epochs = n_epochs,
                      batch = bs,
                      device = gpu_id,
                      patience = waitNum,
                      val = validate,
                      workers = workerNum,)
#config file draws in the path and training data along with the validation info
#epochs in number of training phases so more the better  Estimation = 100-300