import os
import glob
import yaml
import shutil
import random
from collections import defaultdict

def organize_imgs(yaml_dir, imgs_dir):
    """Organize images according to their yaml file classes. Creates directory for each animal class
        and moves the corresponding images in that directory
    
    Args:
        - yaml_dir (str): directory with all .yaml files of detections
        - imgs_dir (str): directory of all the images
    
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
                img_paths = [os.path.join(imgs_dir, fname['file']) for fname in yaml_data['images']]
                for img in img_paths:
                    shutil.copy2(img, folder_path)
                # after operations on imgs are finished, move yaml in folder, too
                shutil.move(file_path, folder_path)
                

def rename_files(imgs_dir):
    """Simple utils to copy the images with needed info to a common directory

    Args:
        imgs_dir (str): parent directory of all species directories
    """
    # make sure dst directory exists
    alldir = os.path.join(imgs_dir, "all")
    os.makedirs(alldir, exist_ok=True)

    for dirname in os.listdir(imgs_dir):
        if dirname == "all" or dirname == "Empty":
            continue
        print(dirname)
        imgs = glob.glob(os.path.join(imgs_dir, dirname, "*.JPG"))
        #print(imgs)
        for img in imgs:
            shutil.copy(img, os.path.join(alldir, f"{dirname}_{os.path.basename(img)}"))
            

def random_subsample(imgs_dir, sample_dim=200):
    """Function to get a random subsample of images, starting from the directory structure (considering also # of imgs for each species)

    Args:
        imgs_dir (str): Parent directory containing all the subfolders for the species
        sample_dim (int, optional): Number of sample images to extract. Defaults to 200.
    """
    # get all images from the "all" directory
    all_imgs_dir = os.path.join(imgs_dir, "all")
    if not os.path.exists(all_imgs_dir):
        print(f"Directory {all_imgs_dir} does not exist. Please run rename_files first.")
        return
    
    # get all JPG files
    all_images = glob.glob(os.path.join(all_imgs_dir, "*.JPG"))
    
    # get existing train image filenames 
    traindir = os.path.join(imgs_dir, "SAMPLE300")
    train_filenames = set()
    
    if os.path.exists(traindir):
        train_filenames = {os.path.basename(img) for img in glob.glob(os.path.join(traindir, "*.JPG"))}
    
    print(f"Found {len(train_filenames)} images in train set")

    # extract species names and group images by species
    species_images = defaultdict(list)
    for img_path in all_images:
        filename = os.path.basename(img_path)
        splittext = "_IM" if "IM" in filename else "_DSCF"
        # get species name (everything before the first underscore)
        species = filename.split(splittext)[0]
        # check overlap with train set by comparing filenames
        if filename not in train_filenames:
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
        if len(species_images[species]) < image_counts[species]+1:
            image_counts[species] = -999    # set fixed value to discard later 
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
    
    # Copy selected images to destination
    for img_path in selected_images:
        shutil.copy2(img_path, subsample_dir)
    
    print(f"Created subsample with {len(selected_images)} images in {subsample_dir}")
    
    # Print distribution summary
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




if __name__ == "__main__":
    # organize_imgs("/home/dario/Projects/eu_wildlife_data/md/MOF_species", "/home/dario/Projects/eu_wildlife_data/img")
    
    # rename_files("/home/dario/Projects/eu_wildlife_data/md/BNP_species")

    # random_subsample("/home/dario/Projects/eu_wildlife_data/md/BNP_species", sample_dim=300)

    pass