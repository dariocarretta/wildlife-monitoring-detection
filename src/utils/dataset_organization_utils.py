import glob
import os
import random
import shutil
from collections import defaultdict
from typing import List

import yaml


def organize_imgs(yaml_dir: str, imgs_dir: str) -> None:
    """Organize images according to their yaml file classes. Creates directory for each animal class
        and moves the corresponding images in that directory.

    Args:
        - yaml_dir (str): directory with all .yaml files of detections.
        - imgs_dir (str): directory of all the images.

    """
    for filename in os.listdir(yaml_dir):
        if filename.endswith(".yaml"):
            file_path = os.path.join(yaml_dir, filename)
            # create a folder with corresponding file name (to put yaml & imgs)
            folder_path = os.path.join(yaml_dir, os.path.splitext(file_path)[0])
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            with open(file_path, "r") as f:
                yaml_data = yaml.safe_load(f)
                img_paths = [
                    os.path.join(imgs_dir, fname["file"])
                    for fname in yaml_data["images"]
                ]
                for img in img_paths:
                    shutil.copy2(img, folder_path)
                # after operations on imgs are finished, move yaml in folder, too
                shutil.move(file_path, folder_path)


def rename_files(imgs_dir: str) -> None:
    """Simple util to copy the images with needed info to a common directory.

    Args:
        imgs_dir (str): parent directory of all species directories.
    """
    # make sure dst directory exists
    alldir = os.path.join(imgs_dir, "all")
    os.makedirs(alldir, exist_ok=True)

    for dirname in os.listdir(imgs_dir):
        if dirname == "all" or dirname == "Empty":
            continue
        print(dirname)
        imgs = glob.glob(os.path.join(imgs_dir, dirname, "*.JPG"))
        # print(imgs)
        for img in imgs:
            shutil.copy(img, os.path.join(alldir, f"{dirname}_{os.path.basename(img)}"))


def random_subsample(
    imgs_dir: str, sample_dim: int = 300, exclude_dirs: List[str] = None
) -> None:
    """Function to get a random subsample of images, starting from the directory structure (considering also # of imgs for each species)

    Args:
        imgs_dir (str): Parent directory containing all the subfolders for the species.
        sample_dim (int, optional): Number of sample images to extract. Defaults to 300.
        exclude_dirs (list, optional): List of (relative) directory names to exclude from sampling. Defaults to ["train_dir"].
    """
    # get all images from the "all" directory
    all_imgs_dir = os.path.join(imgs_dir, "all")
    if not os.path.exists(all_imgs_dir):
        print(
            f"Directory {all_imgs_dir} does not exist. Please run rename_files first."
        )
        return

    # get all JPG files
    all_images = glob.glob(os.path.join(all_imgs_dir, "*.JPG"))

    # get existing image filenames from exclusion directories
    if exclude_dirs is None:
        exclude_dirs = ["SAMPLE300"]

    excluded_filenames = set()
    for exclude_dir in exclude_dirs:
        exclude_path = os.path.join(imgs_dir, exclude_dir)
        if os.path.exists(exclude_path):
            excluded_files = {
                os.path.basename(img)
                for img in glob.glob(os.path.join(exclude_path, "*.JPG"))
            }
            excluded_filenames.update(excluded_files)
            print(f"Found {len(excluded_files)} images in {exclude_dir}")
        else:
            print(
                f"Warning: Exclude directory '{exclude_dir}' does not exist - no overlap checking for this directory"
            )

    print(f"Total excluded images: {len(excluded_filenames)}")

    # extract species names and group images by species
    species_images = defaultdict(list)
    for img_path in all_images:
        filename = os.path.basename(img_path)
        splittext = "_IM" if "IM" in filename else "_DSCF"
        # get species name (everything before the first underscore)
        species = filename.split(splittext)[0]
        # check overlap with excluded sets by comparing filenames
        if filename not in excluded_filenames:
            species_images[species].append(img_path)

    # shuffle images for each species to later use random extraction
    for species in species_images:
        random.shuffle(species_images[species])

    # create destination directory
    subsample_dir = os.path.join(imgs_dir, f"SAMPLE{sample_dim}")
    os.makedirs(subsample_dir, exist_ok=True)

    # balanced image sampling
    selected_images = []
    species_list = list(species_images.keys())
    random.shuffle(species_list)

    species_idx = 0
    image_counts = {species: 0 for species in species_list}

    while len(selected_images) < sample_dim:
        # get current species
        species = species_list[species_idx]

        # skip species with no more elements
        # and set their counts to a fixed value to denote as empty
        if len(species_images[species]) < image_counts[species] + 1:
            image_counts[species] = -999  # set fixed value to discard later
            species_idx = (species_idx + 1) % len(species_list)
            if species_idx == 0:
                # remove species with no more elements
                species_list = [el for el in species_list if image_counts[el] >= 0]
            continue

        print(species)
        print(f"Number of images: {len(species_images[species])}")
        print(image_counts[species])
        selected_images.append(species_images[species][image_counts[species]])
        image_counts[species] += 1

        # get element from new species:
        # the index cycles when we have reached the end of the list, coming back to index 0
        species_idx = (species_idx + 1) % len(species_list)
        # when we reach the end of the list, and come back
        if species_idx == 0:
            # remove species with no more elements
            species_list = [el for el in species_list if image_counts[el] >= 0]

    # copy selected images to destination
    for img_path in selected_images:
        shutil.copy2(img_path, subsample_dir)

    print(f"Created subsample with {len(selected_images)} images in {subsample_dir}")

    # print distribution summary
    species_count = defaultdict(int)
    for img_path in selected_images:
        filename = os.path.basename(img_path)
        if "_IM" in filename:
            species = filename.split("_IM")[0]
        elif "DSCF" in filename:
            species = filename.split("_DSCF")[0]
        species_count[species] += 1

    print("Species distribution in subsample:")
    for species, count in sorted(species_count.items()):
        total_available = len(species_images[species])
        print(f"  {species}: {count}/{total_available} images")


def organize_split_data(
    source_dir: str, split_type: str, output_dir: str = None
) -> None:
    """Organize image-label pairs into folders ready for detection (train/test/val splits).

    Args:
        source_dir (str): Directory containing images (.JPG) and labels (.txt).
        split_type (str): Type of split ('train', 'test', or 'val').
        output_dir (str, optional): Output directory. If None, uses source_dir parent.
    """
    if split_type not in ["train", "test", "val"]:
        raise ValueError("split_type must be 'train', 'test', or 'val'")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(source_dir), "dataset_splits")

    images_dir = os.path.join(output_dir, "images", split_type)
    labels_dir = os.path.join(output_dir, "labels", split_type)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # find image-label pairs
    image_files = glob.glob(os.path.join(source_dir, "*.JPG"))
    valid_pairs = [
        (
            img,
            os.path.join(
                source_dir, f"{os.path.splitext(os.path.basename(img))[0]}.txt"
            ),
        )
        for img in image_files
        if os.path.exists(
            os.path.join(
                source_dir, f"{os.path.splitext(os.path.basename(img))[0]}.txt"
            )
        )
    ]

    valid_pairs.sort(key=lambda x: os.path.basename(x[0]))

    # copy and rename files
    for i, (img_path, label_path) in enumerate(valid_pairs, 1):
        shutil.copy2(img_path, os.path.join(images_dir, f"{split_type}_img{i}.JPG"))
        shutil.copy2(label_path, os.path.join(labels_dir, f"{split_type}_img{i}.txt"))

    print(f"Organized {len(valid_pairs)} pairs into {split_type} split at {output_dir}")


if __name__ == "__main__":
    # organize_imgs("/home/dario/Projects/eu_wildlife_data/md/MOF_species", "/home/dario/Projects/eu_wildlife_data/img")

    # rename_files("/home/dario/Projects/eu_wildlife_data/md/BNP_species")

    # random_subsample("/home/dario/Projects/eu_wildlife_data/md/MOF_species", sample_dim=100, exclude_dirs=["SAMPLE50", "SAMPLE300"])

    organize_split_data(
        "/home/dario/Projects/wildlife-monitoring-detection/european_mammals/train",
        "train",
    )
