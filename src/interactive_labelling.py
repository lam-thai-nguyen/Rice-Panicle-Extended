import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.interactive_labelling.InteractiveLabelling import InteractiveLabelling
from matplotlib import pyplot as plt
from PIL import Image


if __name__ == "__main__":
    img_path = "data/processed/African/1_2_1_1_1_DSC01251.jpg"
    labeler = InteractiveLabelling(img_path=img_path)
    labeler.run()
    labeler.show_update_img()
