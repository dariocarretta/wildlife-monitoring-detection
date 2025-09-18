import os

from ultralytics import YOLO

import wandb

# import albumentations not needed since YOLO integrates automatically with it

# login on wandb
wandb.login(key=os.environ.get("WANDB_API_KEY"))

# load model
model = YOLO("../weights/yolo11n.pt")

# MDV1000: https://github.com/agentmorris/MegaDetector/blob/main/docs/release-notes/mdv1000-release.md

# * NOTE: for the multiclass case, 
# * the weights of the classification and dfl loss were set to 2 and 7, respectively
# * instead of the default values of 0.5 and 1.5


# train (with wandb monitoring)
model.train(
    data="../../data/custom_dataset.yaml",
    epochs=100,
    # ===basic augmentations===
    fliplr=0.5,    # flip horizontally (0.5 prob)
    flipud=0.5,    # flip vertically (0.5 prob)
    hsv_s=0.7,    # alters saturation of image
    hsv_v=0.6,    # alters brightness of image

    imgsz=640,
    batch=16,
    device="cuda",
    workers=8,
    conf=0.2,    # minimum confidence threshold for detections on validation

    iou=0.6,  # iou to consider for non-max suppression

    freeze=10,  # freeze the first 10 layers of the model (the featuer extraction backbone)
    
    project="wildlife-detection",
    name="yolov11n-eumammals-noweighting",
)
# evaluate model
model.val()
