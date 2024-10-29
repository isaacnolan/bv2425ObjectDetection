from ultralytics import YOLO
import os

#load the model
model = YOLO("yolov8m.pt")#loads nano detection YOLO version 8 neural network

n_epochs = 50
bs = -1
gpu_id = 0, 1
imgSize = 640
waitNum = 5
workerNum = os.cpu_count()
OptimizerChoice = 'auto'
validate = True

#Train
results = model.train(data = "data.yaml",
                      imgsz = imgSize,
                      pretrained = True,
                      name = "CharModel1",
                      cos_lr=True,
                      #lr0=0.00269,
                      #lrf=0.00288,
                      lr0=0.00269,
                      lrf=0.01288,
                      epochs = n_epochs,
                      batch = bs,
                      device = gpu_id,
                      patience = waitNum,
                      val = validate,
                      workers = workerNum,)
#config file draws in the path and training data along with the validation info
#epochs in number of training phases so more the better  Estimation = 100-300
