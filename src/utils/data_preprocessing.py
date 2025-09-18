import glob
import os

import cv2
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, List
import numpy as np


class ClassificationDataset(Dataset):
    """Class for dataset created for classification of detected animals."""

    def __init__(
        self,
        parent_dir: str,
        transform: transforms = None,
        split: str = "train",
        fixed_size: Tuple[int, int] = (512, 512),
    ):
        """
        Args:
            parent_dir (str): Base directory of classification dataset.
            transform (transforms, optional): Optional transformations to be applied to the dataset. Defaults to None.
            split (str, optional): Split of the dataset, used to extract corrresponding correct labels. Defaults to "train".
            fixed_size (Tuple[int, int], optional): Standard size to which images are resized. Defaults to (512, 512).
        """
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
        self.fixed_size = fixed_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, int]:
        filename = self.paths[idx]
        img_path = os.path.join(self.img_dir, filename)
        # uncomment for letterbox resizing
        # img = letterbox_resizing(img_path, new_shape=self.fixed_size)
        img = cv2.imread(img_path)
        # resize image to fixed size, e.g., 224x224
        img = cv2.resize(img, self.fixed_size, interpolation=cv2.INTER_LINEAR)
        # convert BGR to RGB for proper color channels
        # (not needed if using letterbox, as it is done in the function)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)

        # apply possible transformations
        if self.transform:
            img = self.transform(img)

        # find the corresponding label for the filename
        label_row = self.labels[self.labels["filename"] == filename]
        label = label_row["class_id"].iloc[0]

        return img, label


def create_dataset_yaml(
    parent_dataset: str, yaml_path: str = "custom_dataset.yaml"
) -> str:
    """Function to create the .yaml config file for YOLO to be used in ultralytics API.

    Args:
        parent_dataset (str): parent_dataset (str): Base dataset containing images and labels folders.
        yaml_path (str, optional): Path to the output .yaml file. Defaults to "custom_dataset.yaml".

    Returns:
        str: prints on the console the content of the created .yaml file.
    """
    with open(os.path.join(parent_dataset, "labels.txt"), "r") as source_classes:
        classes = source_classes.read().strip().split("\n")

    names = {i: cl for i, cl in enumerate(classes)}
    # create string structure as required by the .yaml of yolo
    names_str = "\n ".join(f"{k}: {v}" for k, v in names.items())

    with open(yaml_path, "w+") as f:
        f.write(f"path: {parent_dataset}\ntrain: images/train\nval: images/val\n")
        f.write("train_label_dir: labels/train\nval_label_dir: labels/val\n")
        f.write(f"names:\n {names_str}\n")
        f.seek(0)  # bring "reading cursor" back to beginning
        content = f.read()

    return content


def letterbox_resizing(
    impath: str, new_shape: Tuple[int, int] = (640, 640)
) -> np.array:
    """Letterbox resizing util function, to resize images keeping h-w ratio

    Args:
        impath (str): Path to the image file
        new_shape (tuple, optional): Final dimensions of the resized image. Defaults to (640, 640).

    Returns:
        np.array: Resized image with dimensions (new_shape, new_shape, 3)
    """

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
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    # ensure exact dimensions by checking final size
    if img.shape[:2] != (new_shape[0], new_shape[1]):
        img = cv2.resize(
            img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR
        )

    # convert BGR to RGB for proper color channels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def crop_and_save_detections(
    image_path: str,
    output_dir: str = "detected_animals",
    dataset_path: str = "dataset",
    split: str = "train",
) -> Tuple[int, List[dict]]:
    """Crop detected animals from a single image and save them to a separate directory using YOLO format labels.
    Cropped images are saved as `<src_img_name>_<detection_count>.JPG`.

    Args:
        image_path (srt): Path to the image to be processed.
        output_dir (str, optional): Output directory where cropped images and labels are saved.
                                    Defaults to "detected_animals".
        dataset_path (str, optional): Path to the base input detection directory.
                                    Defaults to "dataset".
        split (str, optional): Split to consider where image should be saved.
                                    Defaults to "train".

    Returns:
        tuple: Tuple containing number of detections in the image and corresponding labels.
    """
    split_output_dir = os.path.join(output_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)

    # load class mappings
    class_mappings_path = os.path.join(dataset_path, "labels.txt")
    class_names = []
    if os.path.exists(class_mappings_path):
        with open(class_mappings_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]

    img = Image.open(image_path)
    img_width, img_height = img.size
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    label_path = os.path.join(dataset_path, "labels", split, f"{img_name}.txt")
    if not os.path.exists(label_path):
        return 0, []

    detection_count = 0
    crop_labels = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # if emtpy label (no bboxes), skip
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])

            # convert to absolute coordinates and calculate bounding box
            x1 = max(0, int((x_center - width / 2) * img_width))
            y1 = max(0, int((y_center - height / 2) * img_height))
            x2 = min(img_width - 1, int((x_center + width / 2) * img_width))
            y2 = min(img_height - 1, int((y_center + height / 2) * img_height))

            # if we have problms with bounding boxes coords, skip
            if x2 <= x1 or y2 <= y1:
                continue

            detection_count += 1
            class_name = (
                class_names[class_id]
                if class_id < len(class_names)
                else f"class_{class_id}"
            )

            # crop and save
            cropped_img = img.crop((x1, y1, x2, y2))
            output_filename = f"{img_name}_{detection_count}.JPG"
            cropped_img.save(os.path.join(split_output_dir, output_filename))

            crop_labels.append(
                {
                    "filename": output_filename,
                    "class_label": class_name,
                    "class_id": class_id,
                    "split": split,
                }
            )

    return detection_count, crop_labels


def process_all_images_from_dataset(
    dataset_path: str = "dataset",
    split: str = "train",
    output_dir: str = "detected_animals",
) -> Tuple[int, list]:
    """Process all images in the detection dataset folder and crop detected animals.

    Args:
        dataset_path (str, optional): Base path of detection dataset directory.
                                    Defaults to "dataset".
        split (str, optional): Name of the split to process.
                                    Defaults to "train".
        output_dir (str, optional): Path to the output classification dataset firectory.
                                    Defaults to "detected_animals".

    Returns:
        tuple: Tuple with number of all crops and labels saved.
    """
    images_dir = os.path.join(dataset_path, "images", split)
    image_files = glob.glob(os.path.join(images_dir, "*JPG"))

    all_labels = []
    total_crops = 0

    for image_path in image_files:
        crops, labels = crop_and_save_detections(
            image_path, output_dir, dataset_path, split
        )
        total_crops += crops
        all_labels.extend(labels)

    return total_crops, all_labels


def process_dataset(
    dataset_path: str = "detection_dataset",
    output_dir: str = "class_dataset",
    csv_filename: str = "class_labels.csv",
) -> Tuple[int, list]:
    """Process train val and test splits to get a classification dataset, and create a common CSV file.

    Args:
        dataset_path (str, optional): Path to the input detection dataset directory.
                                    Defaults to "detection_dataset".
        output_dir (str, optional): Path to the output classification dataset firectory.
                                    Defaults to "class_dataset".
        csv_filename (str, optional): Name of the .csv file with image class data.
                                    Defaults to "class_labels.csv".

    Returns:
        tuple: Tuple with number of all crops and labels saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_labels_combined = []
    total_crops_combined = 0

    for split in ["train", "val", "test"]:
        crops, labels = process_all_images_from_dataset(dataset_path, split, output_dir)
        total_crops_combined += crops
        all_labels_combined.extend(labels)
        print(f"{split}: {crops} crops")

    if all_labels_combined:
        # produce csv file
        df = pd.DataFrame(all_labels_combined)
        df.to_csv(os.path.join(output_dir, csv_filename), index=False)
        print(
            f"Total: {total_crops_combined} crops, {len(all_labels_combined)} labels saved to {csv_filename}"
        )
        print(f"Class distribution:\n{df['class_label'].value_counts()}")

    return total_crops_combined, all_labels_combined


if __name__ == "__main__":
    process_dataset(
        "./european_mammals_dataset/", "./eu_mammals_classification", "class_labels.csv"
    )

    # a = create_dataset_yaml("./european_mammals_dataset/", yaml_path="tmp.yaml")
    # print(a)
