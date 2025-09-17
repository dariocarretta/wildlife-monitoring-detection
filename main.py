import glob
import os
import sys

from ultralytics import YOLO

from src.utils.onestep import (
    plot_detections,
    plot_detector_comparison,
    run_detector_comparison,
)
from src.utils.twostep import (
    load_classifiers,
    plot_two_step_detections,
    run_two_step_detection,
)


def main(input_dir: str, output_dir: str, approach: str = "two-step") -> None:
    """Main function to run test inferences comparing models.

    Args:
        input_dir (str): Directory of input images used for inference.
        output_dir (str): Ouptut directory where comparison images will be saved.
        approach (str, optional): _description_. Defaults to "two-step".

    Raises:
        ValueError: If the chosen approach is not among the available ones.
    """
    if approach not in ["two-step", "one-step", "detector-comparison"]:
        raise ValueError(
            "Approach must be 'two-step', 'one-step', or 'detector-comparison'"
        )

    # get image files
    extensions = [
        "*.JPG",
        "*.jpg",
        "*.JPEG",
        "*.jpeg",
        "*.png",
        "*.PNG",
        "*.bmp",
        "*.webp",
    ]
    data = []
    for ext in extensions:
        data.extend(glob.glob(os.path.join(input_dir, ext)))

    # onestep: direct detection+classification
    if approach == "one-step":
        # load models and run predictions
        model1 = YOLO("./src/weights/one-step/onestep_yolo11n.pt")
        model2 = YOLO("./src/weights/one-step/onestep_mdv1000.pt")

        yolo_results = model1.predict(source=data, conf=0.25, verbose=False)
        md_results = model2.predict(source=data, conf=0.25, verbose=False)

        plot_detections(data, yolo_results, md_results, output_dir)

    # compare  base detectors
    elif approach == "detector-comparison":
        
        megadetector = YOLO("./src/weights/md_v1000.0.0-sorrel.pt")    # base megadetector
        yolo_detector = YOLO("./src/weights/two-step/yolo11n_singlecls.pt")    # finetuned yolo

        comparison_results = run_detector_comparison(data, megadetector, yolo_detector)

        # generate outputs
        plot_detector_comparison(comparison_results, output_dir)

    # twostep: base detection -> classification
    elif approach == "two-step":
        
        detector = YOLO("./src/weights/md_v1000.0.0-sorrel.pt") 

        custom_model, resnet_model, device = load_classifiers()
        
        detections_data = run_two_step_detection(
            data, detector, custom_model, resnet_model, device
        )

        # generate outputs
        plot_two_step_detections(detections_data, output_dir)



if __name__ == "__main__":
    # allow command line arguments for different approaches
    if len(sys.argv) > 1:
        approach = sys.argv[1]
    else:
        approach = "all"  # default to run all approaches

    input_dir = "./presentation_utils/test_images/"
    output_dir = "./results/"

    if approach == "all":
        main(input_dir, output_dir, approach="one-step")
        main(input_dir, output_dir, approach="two-step")
        main(input_dir, output_dir, approach="detector-comparison")
    else:
        main(input_dir, output_dir, approach=approach)