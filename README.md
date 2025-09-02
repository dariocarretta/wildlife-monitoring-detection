# Wildlife Monitoring (Detection)

This is the repository for the final project of my Master's Degree course in Vision for Industry and Environment.
It is centered around an image detection task, specifically focused on wildlife monitoring.

The dataset a subset of [the MOF and BNP datasets](https://data.uni-marburg.de/entities/dataset/eafc2547-4616-48a4-b9ee-cd28f207afba) (licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)), which consists of a total of ~2500 camera trap images, recorded in the Marburg Open Forest in Hesse, Germany and  Podlaskie Voivodeship, Poland [^1]
([Schneider et al., 2024](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/cvi2.12294)).

In particular, I worked on a 300 (train & val) + 75 (test) images subset of the BNP and MOF datasets, and applied manual labelling with the [YoloLabel](https://github.com/developer0hye/Yolo_Label?tab=readme-ov-file) tool on this subset.

[^1]: Schneider, Daniel; Lindner, Kim; Vogelbacher, Markus; Bellafkir, Hicham; Farwig, Nina; Freisleben, Bernd: Recognition of European mammals and birds in camera trap images using deep neural networks. IET Computer VIsion, 2024
