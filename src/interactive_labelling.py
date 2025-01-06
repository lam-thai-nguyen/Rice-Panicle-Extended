import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib import pyplot as plt
from PIL import Image
from scripts.generate_annotations.riceprManager import riceprManager
from scripts.interactive_labelling.InteractiveLabelling import InteractiveLabelling


def interactive_labelling(img_path, save_path="data/processed"):
    # Step 1: Create instance
    labeler = InteractiveLabelling(img_path=img_path, save_path=save_path)

    # Step 2: Interactive Labeling
    labeler.run()

    # Step 3: Show updated image
    labeler.show_update_img()
    

def show_update_img(img_path):
    filename = img_path.split("/")[-1]
    species = img_path.split("/")[-2]
    updated_ricepr = f"data/processed/{species}/{filename.replace('.jpg', '.ricepr')}"
    original_img = Image.open(f"data/raw/{species}/{filename}")
    
    assert updated_ricepr is not None, "Updated ricepr not found!"
    
    ricepr_manager = riceprManager(PATH=updated_ricepr)
    junctions_manager = ricepr_manager.read_ricepr()[0]
    junctions = junctions_manager.return_junctions()
    
    plt.figure(figsize=(10, 9))
    plt.imshow(original_img)
    for x, y in junctions:
        plt.scatter(x, y, marker='o', c="yellow", s=35, alpha=0.5)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    root_dir_african = "data/processed/African"
    root_dir_asian = "data/processed/Asian"

    # Show updated image and exit
    # Uncomment (4 lines) to use the later features
    img_name = ""  # Change this
    if img_name:
        show_update_img(f"data/raw/Asian/{img_name}.jpg")
        exit()

    # African
    for filename in os.listdir(root_dir_african):
        if filename.endswith(".jpg") and not filename.startswith("[done]"):
            img_path = f"{root_dir_african}/{filename}"
            interactive_labelling(img_path)
            print("".center(50, "="))

    # Asian
    for filename in os.listdir(root_dir_asian):
        if filename.endswith(".jpg") and not filename.startswith("[done]"):
            img_path = f"{root_dir_asian}/{filename}"
            interactive_labelling(img_path)
            print("".center(50, "="))
