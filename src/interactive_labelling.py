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
    img_path = "data/processed/African/1_2_1_1_1_DSC01251.jpg"
    interactive_labelling(img_path)
    # show_update_img(img_path)
