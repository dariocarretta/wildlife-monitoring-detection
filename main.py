import torch
import torch.nn as nn
from ultralytics import YOLO

# TODO: complete file with main function to be run, that
# 1. Runs the megadetector to get bounding boxes
# 2. On these bounding boxes runs classification
# 3. gives summary of the process (e.g., 3 detections in image 2: [boar, mouse, cat])
# 4. plots the detctions with the needed info (i.e. source image, class)