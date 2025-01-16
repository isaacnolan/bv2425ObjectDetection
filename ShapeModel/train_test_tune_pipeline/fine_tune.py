#this will be an algorithm that fine tunes a YOLO model on synth data
# https://docs.ultralytics.com/integrations/ray-tune/#custom-search-space-example 
# https://docs.ultralytics.com/guides/hyperparameter-tuning/ 

import wandb
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ray import tune

'''
#TENSORBOARD ADDITIVES
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf

os.system("tensorboard --logdir=summaries")
#https://www.datacamp.com/tutorial/tensorboard-tutorial
'''

storage_path = "tune"
exp_name = "exp_1_17_24"
train_mnist = 0 #change this****
wandb.init(project="YOLO-Tuning", name = exp_name)

# Load YOLO model
model = YOLO("yolo11m.pt")

# Tune hyperparameters
result_grid = model.tune(
    data="data_yaml.yaml", #Make absolute path
    space={"lr0": tune.uniform(1e-5, 1e-1)},
    epochs=50,
    use_ray=True,
    gpu_per_trial=1,
    device=0)

experiment_path = f"{storage_path}/{exp_name}"
print(f"Loading results from {experiment_path}...")

restored_tuner = tune.Tuner.restore(experiment_path, trainable=train_mnist)
result_grid = restored_tuner.get_results()

for i, result in enumerate(result_grid):
    plt.plot(
        result.metrics_dataframe["training_iteration"],
        result.metrics_dataframe["mean_accuracy"],
        label=f"Trial {i}",
    )

plt.xlabel("Training Iterations")
plt.ylabel("Mean Accuracy")
plt.legend()
plt.show()