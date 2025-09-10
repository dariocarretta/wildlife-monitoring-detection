import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import ToTensor
import cv2
from PIL import Image
import glob
import pandas as pd

# TODO: 
# 0. Organize input and label files to match expected YOLO structure ✅
# 1. Create classification dataset class 
# 2. Choose data augmentation functions to apply to training 
# 3. Create custom train test split according to the class distributions (training should have at least one example
#    from all the classes)

# TODO: cleanup function

class ClassificationDataset(Dataset):
    """Class for dataset created for detection
    
    Args:
        parent_dir (str): parent dir containing images and labels dirs
        transform (torchvision.transforms, optional): transformations to be applied to the images. Defaults to None.
        split (str, optional): train or test split, needed to get data. Defaults to "train".
    """
    def __init__(self, parent_dir, transform=None, split="train"):
        super().__init__()
        self.parent_dir = parent_dir
        self.img_dir = os.path.join(parent_dir, split)
        self.paths = os.listdir(self.img_dir)
        self.labels_path = os.path.join(parent_dir, "class_labels.csv")
        self.labels = pd.read_csv(self.labels_path)
        # reset_index=True is needed to get the relative indices of the selected rows
        self.labels = self.labels[self.labels["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        filename = self.paths[index]
        img_path = os.path.join(self.img_dir, filename)
        # apply resizing to common size and convert to torh tensor in [0., 1.]
        img = ToTensor(letterbox_resizing(img_path, new_shape=(512, 512)))
        # apply possible transformations
        if self.transform:
            img = self.transform(img)
            
        # find the corresponding label for the filename
        label_row = self.labels[self.labels['filename'] == filename]
        label = label_row['class_id'].iloc[0]

        return img, label




def create_dataset_yaml():
    with open("./dataset/class_mappings.txt", "r") as source_classes:
        classes = source_classes.read().strip().split("\n")

    names = {i: cl for i, cl in enumerate(classes)}
    # create string structure as required by the .yaml of yolo
    names_str = "\n ".join(f"{k}: {v}" for k, v in names.items())

    with open("custom_dataset.yaml", "w") as f:
        f.write("path: ./dataset\ntrain: images/train\nval: images/test\n")
        f.write("train_label_dir: labels/train\nval_label_dir: labels/test\n")
        f.write(f"names:\n {names_str}\n")


def letterbox_resizing(impath, new_shape=(640, 640)):
    """Letterbox resizing util function, to resize images keeping h-w ratio

    Args:
        impath (str): path to the image file
        new_shape (tuple, optional): Final dimensions of the resized image. Defaults to (640, 640).

    Returns:
        np.array: resized image 
    """

    # load image
    img = cv2.imread(impath)
    
    shape = img.shape[:2]  # current shape [height, width]
    # calculate scale ratio 
    # (1.0 is for images that are already of right dimensions, to keep them as they are)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1], 1.0)
    # calculate new dimensions after scaling
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    
    # resize image
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # calculate padding needed
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    # split padding between both sides (with 0.1 element to handle odd padding split in 2)
    top, bottom = int(round(dh // 2 - 0.1)), int(round(dh // 2 + 0.1))
    left, right = int(round(dw // 2 - 0.1)), int(round(dw // 2 + 0.1))
    
    # add padding with gray color
    color = (114, 114, 114)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img


def crop_and_save_detections(image_path, output_folder="detected_animals", dataset_backup_path="dataset_backup", split="train"):
    """Crop detected animals from image and save them to a separate folder using YOLO format labels"""
    split_output_folder = os.path.join(output_folder, split)
    os.makedirs(split_output_folder, exist_ok=True)
    
    # Load class mappings
    class_mappings_path = os.path.join("detection_dataset", "class_mappings.txt")
    class_names = []
    if os.path.exists(class_mappings_path):
        with open(class_mappings_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    
    img = Image.open(image_path)
    img_width, img_height = img.size
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    label_path = os.path.join(dataset_backup_path, "labels", split, f"{img_name}.txt")
    if not os.path.exists(label_path):
        return 0, []
    
    detection_count = 0
    crop_labels = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # Convert to absolute coordinates and calculate bounding box
            x1 = max(0, int((x_center - width/2) * img_width))
            y1 = max(0, int((y_center - height/2) * img_height))
            x2 = min(img_width-1, int((x_center + width/2) * img_width))
            y2 = min(img_height-1, int((y_center + height/2) * img_height))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            detection_count += 1
            class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
            
            # Crop and save
            cropped_img = img.crop((x1, y1, x2, y2))
            output_filename = f"{img_name}_{detection_count}.JPG"
            cropped_img.save(os.path.join(split_output_folder, output_filename))
            
            crop_labels.append({
                'filename': output_filename,
                'class_label': class_name,
                'class_id': class_id,
                'split': split
            })
    
    return detection_count, crop_labels


def process_all_images_from_dataset(dataset_path="dataset", split="train", output_folder="detected_animals"):
    """Process all images in the detection dataset folder and crop detected animals"""
    images_dir = os.path.join(dataset_path, "images", split)
    image_files = glob.glob(os.path.join(images_dir, "*JPG"))
    
    all_labels = []
    total_crops = 0
    
    for image_path in image_files:
        crops, labels = crop_and_save_detections(image_path, output_folder, dataset_path, split)
        total_crops += crops
        all_labels.extend(labels)
    
    return total_crops, all_labels


def process_dataset(dataset_path="detection_dataset", output_folder="class_dataset", csv_filename="class_labels.csv"):
    """Process both train and test splits and create a common CSV file"""
    os.makedirs(output_folder, exist_ok=True)
    
    all_labels_combined = []
    total_crops_combined = 0
    
    for split in ["train", "test"]:
        crops, labels = process_all_images_from_dataset(dataset_path, split, output_folder)
        total_crops_combined += crops
        all_labels_combined.extend(labels)
        print(f"{split}: {crops} crops")
    
    if all_labels_combined:
        df = pd.DataFrame(all_labels_combined)
        df.to_csv(os.path.join(output_folder, csv_filename), index=False)
        print(f"Total: {total_crops_combined} crops, {len(all_labels_combined)} labels saved to {csv_filename}")
        print(f"Class distribution:\n{df['class_label'].value_counts()}")
    
    return total_crops_combined, all_labels_combined



if __name__ == "__main__":
    # process_dataset("detection_dataset", "class_dataset", "class_labels.csv")
    ds = ClassificationDataset("class_dataset", split="train")

    print(ds[0])
    