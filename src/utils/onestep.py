import os
from pathlib import Path
from typing import List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results


def run_detector_comparison(
    image_paths: List[str], megadetector: YOLO, yolo_detector: YOLO
) -> List[dict]:
    """Compare base MegaDetector vs YOLO single-class detector.

    Args:
        image_paths (List[str]): List of image paths to do inference on.
        megadetector (YOLO): Base MegaDetector model to compare the finetuned yolo against.
        yolo_detector (YOLO): Finetuned YOLO model.

    Returns:
        List[dict]: The list of all results (detections+classifications) for both detector models.
    """
    comparison_results = []

    for image_path in image_paths:
        # run MegaDetector
        md_results = megadetector.predict(source=image_path, conf=0.25, verbose=False)

        # run YOLO single-class detector
        yolo_results = yolo_detector.predict(
            source=image_path, conf=0.25, verbose=False
        )

        result = {"image_path": image_path, "md_detections": [], "yolo_detections": []}

        # extract MegaDetector detections
        if md_results[0].boxes is not None:
            for box in md_results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box
                result["md_detections"].append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": float(conf),
                        "class": "animal",
                    }
                )

        # extract YOLO detections
        if yolo_results[0].boxes is not None:
            for box in yolo_results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box
                result["yolo_detections"].append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": float(conf),
                        "class": "animal",
                    }
                )

        comparison_results.append(result)

    return comparison_results


def plot_detector_comparison(comparison_results: List[dict], output_dir: str) -> None:
    """Plot comparison between MegaDetector and YOLO single-class detector.

    Args:
        comparison_results (List[dict]): The list of all obtained results (detections+classifications) for both detector models.
        output_dir (str): The output directory where results are saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for result in comparison_results:
        img = Image.open(result["image_path"])
        image_path = result["image_path"]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        ax.set_title(
            f"Detector Comparison: {Path(image_path).name}",
            fontsize=14,
            fontweight="bold",
        )

        # plot yolo detections (red, solid line)
        for detection in result["yolo_detections"]:
            x1, y1, x2, y2 = detection["bbox"]
            conf = detection["confidence"]

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="-",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 5,
                f"YOLO: {conf:.2f}",
                color="red",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # plot megadetector detections (blue, dashed line)
        for detection in result["md_detections"]:
            x1, y1, x2, y2 = detection["bbox"]
            conf = detection["confidence"]

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="blue",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)
            ax.text(
                x2,
                y1 - 5,
                f"MDv1000: {conf:.2f}",
                color="blue",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # add legend
        legend_elements = [
            Line2D([0], [0], color="red", lw=2, label="YOLO111n Finetuned"),
            Line2D(
                [0],
                [0],
                color="blue",
                lw=2,
                linestyle="--",
                label="MDv1000",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
        ax.axis("off")

        # save plot
        output_path = os.path.join(
            output_dir, f"detector_comparison_{Path(image_path).stem}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_detections(
    image_paths: List[str], yolo_results: Results, md_results: Results, output_dir: str
) -> None:
    """Plot images with detection results from both models with different colors

    Args:
        image_paths (List[str]): THe list of image paths to do inference on.
        yolo_results (Results): The `Results` object generated by the YOLO model.
        md_results (Results): The `Results` object generated by the MegaDetector model.
        output_dir (str): Output directory where images are saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        ax.set_title(
            f"Detection Comparison: {Path(image_path).name}",
            fontsize=14,
            fontweight="bold",
        )

        # plot yolo detections (red, solid line)
        if i < len(yolo_results) and yolo_results[i].boxes is not None:
            for box in yolo_results[i].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box
                class_name = yolo_results[i].names[int(cls)]

                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    x1,
                    y1 - 5,
                    f"YOLO: {class_name} ({conf:.2f})",
                    color="red",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

        # plot megadetector detections (blue, dashed line)
        if i < len(md_results) and md_results[i].boxes is not None:
            for box in md_results[i].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box
                class_name = md_results[i].names[int(cls)]

                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="blue",
                    facecolor="none",
                    linestyle="--",
                )
                ax.add_patch(rect)
                ax.text(
                    x2,
                    y1 - 5,
                    f"MD: {class_name} ({conf:.2f})",
                    color="blue",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

        # add legend
        legend_elements = [
            Line2D([0], [0], color="red", lw=2, label="YOLO Model"),
            Line2D(
                [0], [0], color="blue", lw=2, linestyle="--", label="MegaDetector Model"
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
        ax.axis("off")

        # save plot
        output_path = os.path.join(
            output_dir, f"onestep_comparison_{Path(image_path).stem}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
