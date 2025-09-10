import  os
from ultralytics import YOLO
import torch.nn as nn
import torch
import wandb
import torchvision

# login on wandb
wandb.login(key=os.environ.get("WANDB_API_KEY"))

#import albumentations not needed since YOLO integrates automatically with it

# reload model from checkpoint
model = YOLO(f"./weights/md_v1000.0.0-sorrel.pt")

# MDV1000: https://github.com/agentmorris/MegaDetector/blob/main/docs/release-notes/mdv1000-release.md

# train (with wandb monitoring)
model.train(
    data="./custom_dataset.yaml", 
    epochs=100,
    # Existing augmentations:
    fliplr=0.5, flipud=0.5, hsv_s=0.7, hsv_v=0.6,
    
    # ADD THESE - all native Ultralytics:
    mosaic=1.0,     # mosaic augmentation (very effective)
    mixup=0.1,      # mixup for better generalization
    copy_paste=0.1, # copy-paste objects between images
    degrees=10,     # rotation augmentation
    translate=0.1,  # translation augmentation
    scale=0.5,      # scaling variation
    
    imgsz=960, batch=8, device=0,
    workers=8,
    conf=0.2,

    single_cls=True,
    freeze=10, # freeze the first 10 layers of the model (the backbone)
    
    project="wildlife-detection", name="megadetectorv1000-frozen-singleclass"
)

# evaluate model (validation results are logged to a W&B table)
model.val()



