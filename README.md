# Wildlife Monitoring (Detection)

This is the repository for the final project of a course in Computer Vision for Industry and Environment from my Master's Degree.
It is centered around an image detection task, specifically focused on wildlife monitoring.

The goal was to analyze different DL methods for wildlife detection that can be used with small datasets, and compare their performance.

## Data

The dataset considered was a subset of [the MOF and BNP datasets](https://data.uni-marburg.de/entities/dataset/eafc2547-4616-48a4-b9ee-cd28f207afba) (licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)), which consist of a total of ~17500 camera trap images, recorded in the Marburg Open Forest in Hesse, Germany and  Podlaskie Voivodeship, Poland [^1]
([Schneider et al., 2024](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/cvi2.12294)).

In particular, I worked on a 305 images subset of the MOF dataset for train, and 70  images taken from the BNP dataset (and the "Empty" directory of the MOF dataset, with non-detected images) for validation/testing. To this subset I applied manual labelling using the [YoloLabel](https://github.com/developer0hye/Yolo_Label?tab=readme-ov-file) detection labeling tool.

The species of animals considered, given the small dimension of the dataset, were simply three, chosen from common European medium-large mammals: 
- Roe Deer (*Capreolus Capreolus*): 109 images included in the training set
- Wild Boar (*Sus Scrofa*): 104 images included in the training set 
- Red Fox (*Vulpes Vulpes*): 87 images included in the trianing set

[To avoid the training of models that would always detect bounding boxes, also 5 no-animals images were included in the training set.]

## Methodologies

The approach I used can be split into four different methodologies, to be compared:
1. Fine-tune a YOLO object detection model (specifically, [YOLOv11n](https://docs.ultralytics.com/models/yolo11/#performance-metrics)) for a direct animal detection + classification in one step.
2. Analogously to approach 1., but by fine-tuning a [MegaDetector](https://github.com/agentmorris/MegaDetector?tab=readme-ov-file#whats-megadetector-all-about) model (specifically, [MDv1000](https://github.com/agentmorris/MegaDetector/blob/main/docs/release-notes/mdv1000-release.md)), which is a SOTA model for animal detection, to also perform species classification in one step.
3. Apply a two-step detection, by using a pretrained MDv1000 to perform animal detection, and finetune [ResNet34](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html) on the resulting crops
4. Apply a two-step detection approach, analogous to 2., but training from scratch a custom classifier (based on [Squeeze-and-Excitation Blocks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)) to run on the resulting detected animals.


### One-Step  Detection: Finetuning YOLOv11n and MegaDetector

![one-step comparison](/media/onestep_comparison.png)

The one-step detection YOLOv11n model was finetuned by:
- Freezing its first 10 layers (the feature extracting backbone)
- Giving larger weights to the classification and [Distribution Focal Loss](https://arxiv.org/pdf/2006.04388) functions, to get more precise classification performances and better predict the probability distribution of the target bounding boxes, focusing on difficult examples 
- Applying data augmentation to reduce overfitting and improve generalization, namely:
    - *Random horizontal flipping*
    - *Random vertical flipping*
    - *Random saturation change*
    - *Random brightness change*
    - [Albumentations](https://albumentations.ai/) data augmentations (*[blur](https://docs.ultralytics.com/integrations/albumentations/#blur), [median blur](https://docs.ultralytics.com/integrations/albumentations/#median-blur), [grayscale conversion](https://docs.ultralytics.com/integrations/albumentations/#grayscale), [CLAHE](https://docs.ultralytics.com/integrations/albumentations/#contrast-limited-adaptive-histogram-equalization-clahe)*), applied automatically during training with the [Ultralytics API integration](https://docs.ultralytics.com/integrations/albumentations/#how-to-use-albumentations-to-augment-data-for-yolo11-training).

The selected image size was 640x640 (as inidicated in YOLOv11n documentation), and the chosen batch size was of 16.

The performances of the fine-tuned YOLOv11n and MDv1000 on the one-step detection + classification task were comparable in accuracy, although with a different balance among the classes:

<p float="middle">
  <img src="/media/onestep_mdv1000.png" width="400" />
  <img src="/media/onestep_yolov11n.png" width="400" /> 
</p>

*Normalized confusion marices for the finetuned MDv1000 (left) and YOLOv11n (right) models. The fine-tuned YOLOv11n shows a more balanced performance across the classes and an overall better one w.r.t. accuracy (although worse on roe deers).*

## Two-Step Detection 

![two-step comparison](/media/twostep_comparison.png)

### Custom Classifier



*Confusion matrices for the finetuned ResNet34 (left) and custom classifier (right) models. It can be seen how ResNet34 has an overall better performance, but not overwhelmingly superior to the custom model.*

## Conclusions











[^1]: Schneider, Daniel; Lindner, Kim; Vogelbacher, Markus; Bellafkir, Hicham; Farwig, Nina; Freisleben, Bernd: Recognition of European mammals and birds in camera trap images using deep neural networks. IET Computer VIsion, 2024
