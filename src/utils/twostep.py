import os
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from ..models.custom_classifier import CustomClassifier


def load_classifiers() -> Tuple[nn.Module, nn.Module, str]:
    """Load the custom and ResNet classifiers for the two-step detection+classification approach.

    Returns:
        Tuple[nn.Module, nn.Module, str]: The custom and finetuned model, and the inference device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load custom classifier
    custom_model = CustomClassifier(num_classes=3, init_features=32)
    custom_model.load_state_dict(
        torch.load("./src/weights/two-step/custom_classifier.pt")
    )
    custom_model.to(device)
    custom_model.eval()

    # load ResNet classifier
    resnet_model = torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet34", pretrained=False
    )
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 3)
    resnet_model.load_state_dict(
        torch.load("./src/weights/two-step/resnet34_classifier.pt")
    )
    resnet_model.to(device)
    resnet_model.eval()

    return custom_model, resnet_model, device


def classify_crops(
    crops: List[Image.Image],
    custom_model: nn.Module,
    resnet_model: nn.Module,
    device: str = "cuda",
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Classify cropped detection images using both classifiers.

    Args:
        crops (List[Image.Image]): List of cropped images after detection.
        custom_model (nn.Module): Custom classifier pytorch model.
        resnet_model (nn.Module): ResNet34 finetuned model in pytorch.
        device (str, optional): Device on which inference is performed. Defaults to "cuda".

    Returns:
        Tuple[List, List]: Tuple of the predicted class names and confidences for custom and finetuned models.
    """
    # if the image has no detections (e.g., image with no animals)
    if not crops:
        return [], []

    # define needed transforms to match training preprocessing exactly
    # only normalization - resizing will be done manually with cv2 to match training
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # class names
    class_names = ["roe deer", "wild boar", "red fox"]

    custom_predictions = []
    resnet_predictions = []

    with torch.no_grad():
        for crop in crops:
            # manual preprocessing to match testing processing
            # 1. convert PIL to numpy array
            img_array = np.array(crop)

            # 2. resize using cv2 with INTER_LINEAR (same as training)
            img_resized = cv2.resize(
                img_array, (224, 224), interpolation=cv2.INTER_LINEAR
            )

            # 3. convert to tensor (this also converts to [0,1] range)
            to_tensor = transforms.ToTensor()
            img_tensor = to_tensor(img_resized)

            # 4. apply normalization (same as training)
            input_tensor = transform(img_tensor).unsqueeze(0).to(device)

            # custom classifier prediction
            custom_output = custom_model(input_tensor)
            custom_pred = torch.softmax(custom_output, dim=1)
            custom_class_idx = torch.argmax(custom_pred, dim=1).item()
            custom_conf = custom_pred[0, custom_class_idx].item()
            custom_predictions.append((class_names[custom_class_idx], custom_conf))

            # ResNet classifier prediction
            resnet_output = resnet_model(input_tensor)
            resnet_pred = torch.softmax(resnet_output, dim=1)
            resnet_class_idx = torch.argmax(resnet_pred, dim=1).item()
            resnet_conf = resnet_pred[0, resnet_class_idx].item()
            resnet_predictions.append((class_names[resnet_class_idx], resnet_conf))

    return custom_predictions, resnet_predictions


def run_two_step_detection(
    image_paths: List[str],
    detector: YOLO,
    custom_model: nn.Module,
    resnet_model: nn.Module,
    device: str = "cuda",
) -> List[dict]:
    """Run two-step detection:
    1) Detect animals
    2) Classify detected regions


    Args:
        image_paths (List[str]): List of the image paths to run the test on.
        detector (YOLO): Finetuned or base MegaDetector YOLO object to run detections.
        custom_model (nn.Module): Custom classifier pytorch model.
        resnet_model (nn.Module): Finetuned ResNet model in pytorch.
        device (str, optional): Device on which inference is performed. Defaults to "cuda".

    Returns:
        List[dict]: The list of result dictionaries for all the input images.

        Each dictionary contains `detections`, `custom_predictions` and `resnet_predictions`.
    """
    all_detections = []

    for image_path in image_paths:
        img = Image.open(image_path)

        # step 1: run detector to get bounding boxes
        results = detector.predict(source=image_path, conf=0.25, verbose=False)

        image_detections = {
            "image_path": image_path,
            "original_image": img,
            "detections": [],
            "custom_predictions": [],
            "resnet_predictions": [],
        }

        if results[0].boxes is not None:
            boxes = results[0].boxes.data.cpu().numpy()

            # step 2: crop detected regions and classify them
            crops = []
            detection_info = []

            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # crop the detection
                crop = img.crop((x1, y1, x2, y2))
                crops.append(crop)
                detection_info.append(
                    {"bbox": (x1, y1, x2, y2), "detection_conf": conf}
                )

            # classify all crops
            custom_preds, resnet_preds = classify_crops(
                crops, custom_model, resnet_model, device=device
            )

            image_detections["detections"] = detection_info
            image_detections["custom_predictions"] = custom_preds
            image_detections["resnet_predictions"] = resnet_preds

        all_detections.append(image_detections)

    return all_detections


def plot_two_step_detections(detections_data: List[dict], output_dir: str) -> None:
    """Plot images with two-step detection results.

    Args:
        detections_data (List[dict]): The list of result dictionaries of the two-step detection process.
        output_dir (str): The output directory where resulting images are saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for img_data in detections_data:
        img = img_data["original_image"]
        image_path = img_data["image_path"]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        ax.set_title(
            f"Two-Step Detection: {Path(image_path).name}",
            fontsize=14,
            fontweight="bold",
        )

        for i, detection in enumerate(img_data["detections"]):
            x1, y1, x2, y2 = detection["bbox"]
            det_conf = detection["detection_conf"]

            # draw detection bbox
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="green",
                facecolor="none",
            )
            ax.add_patch(rect)

            # get classification results
            custom_class, custom_conf = img_data["custom_predictions"][i]
            resnet_class, resnet_conf = img_data["resnet_predictions"][i]

            # add labels
            # custom model
            ax.text(
                x1,
                y1 - 50,
                f"Custom: {custom_class} ({custom_conf:.2f})",
                color="red",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            # finetuned resnet
            ax.text(
                x1,
                y1 - 10,
                f"ResNet: {resnet_class} ({resnet_conf:.2f})",
                color="blue",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            # detections
            ax.text(
                x1,
                y2 + 5,
                f"Det: {det_conf:.2f}",
                color="green",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

        # add legend
        legend_elements = [
            Line2D([0], [0], color="green", lw=2, label="Detection"),
            Line2D([0], [0], color="red", lw=2, label="Custom Classifier"),
            Line2D([0], [0], color="blue", lw=2, label="ResNet Classifier"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
        ax.axis("off")

        # save plot
        output_path = os.path.join(
            output_dir, f"two_step_detection_{Path(image_path).stem}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
