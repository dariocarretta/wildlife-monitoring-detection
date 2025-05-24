import os
import yaml
import shutil

def organize_imgs(yaml_dir, imgs_dir):
    """
    Organize images according to their yaml file classes
    
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
                
            
            
            
if __name__ == "__main__":
    organize_imgs("/home/dario/Projects/eu_wildlife_data/md/BNP_species", "/home/dario/Projects/eu_wildlife_data/img")