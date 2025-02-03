import os 
import shutil


def duplicate_split(src, dst):
    """
    Duplicate a split dataset: same images but different labels

    Args:
        src (str): src split name
        dst (str): dst split name
        
    N.B. Only run this function after the buffer is ready. The buffer is buffer/. Ready means buffer/ contains all and only labels files. To do that, refer to scripts/utils and run those files as modules.
    """
    os.makedirs(f"data/splits/{dst}/train/images", exist_ok=True)
    os.makedirs(f"data/splits/{dst}/train/labels", exist_ok=True)
    os.makedirs(f"data/splits/{dst}/val/images", exist_ok=True)
    os.makedirs(f"data/splits/{dst}/val/labels", exist_ok=True)
    
    buffer = "buffer"

    for filename in os.listdir(buffer):
        new_filename = filename.replace("_junctions.txt", ".txt")  # Rename
        img_name = new_filename.replace(".txt", ".jpg")
        
        # Labels
        if os.path.exists(f"data/splits/{src}/train/labels/{new_filename}"):
            shutil.copy(f"{buffer}/{filename}", f"data/splits/{dst}/train/labels/{new_filename}")
        if os.path.exists(f"data/splits/{src}/val/labels/{new_filename}"):
            shutil.copy(f"{buffer}/{filename}", f"data/splits/{dst}/val/labels/{new_filename}")

        # Images
        if os.path.exists(f"data/splits/{src}/train/images/{img_name}"):
            shutil.copy(f"data/splits/{src}/train/images/{img_name}", f"data/splits/{dst}/train/images/{img_name}")
        if os.path.exists(f"data/splits/{src}/val/images/{img_name}"):
            shutil.copy(f"data/splits/{src}/val/images/{img_name}", f"data/splits/{dst}/val/images/{img_name}")
        
    # Copy and modify the data.yaml file
    yaml_src = f"data/splits/{src}/data.yaml"
    yaml_dst = f"data/splits/{dst}/data.yaml"

    if os.path.exists(yaml_src):
        with open(yaml_src, 'r') as file:
            content = file.read()

        # Replace occurrences of the source split name with the destination split name
        updated_content = content.replace(f"data/splits/{src}", f"data/splits/{dst}")

        # Write the updated content to the new location
        with open(yaml_dst, 'w') as file:
            file.write(updated_content)
    else:
        print(f"data.yaml not found in {yaml_src}")
        
        
if __name__ == "__main__":
    src = "split0"  # Change this if needed
    dst = "split1"  # Change this if needed
    duplicate_split(src, dst)