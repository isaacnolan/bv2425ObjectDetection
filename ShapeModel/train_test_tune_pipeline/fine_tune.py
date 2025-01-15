#this will be an algorithm that fine tunes a YOLO model on synth data
# https://docs.ultralytics.com/integrations/ray-tune/#custom-search-space-example 
# https://docs.ultralytics.com/guides/hyperparameter-tuning/ 

import wandb
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ray import tune

storage_path = "tune"
exp_name = "exp_1_17_24"
train_mnist = 0 #change this****
wandb.init(project="YOLO-Tuning", entity="BuckeyeVertical", name = exp_name)

# Load YOLO model
model = YOLO("yolo11m.pt")

# Tune hyperparameters
result_grid = model.tune(
    data="data_yaml.yaml",
    space={},
    epochs=50,
    use_ray=True)

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