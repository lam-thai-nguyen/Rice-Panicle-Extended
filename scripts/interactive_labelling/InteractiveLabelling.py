"""
Most of the ground truths are not perfect. They are originally the analysis results of P-TRAP 
plus the post-correction by experts. This module is used for interactive labelling. We can click on
the original image to add/remove junctions, which will later be used for annotations generation.

@lam-thai-nguyen #Aug30-2024
"""
from ..generate_annotations.riceprManager import riceprManager
from .ClickHandler import ClickHandler
import matplotlib.pyplot as plt
from PIL import Image
class InteractiveLabelling:
    def __init__(self, img_path):
        self.img_path = img_path
        self.filename = self.img_path.split("/")[-1]
        self.species = self.img_path.split("/")[-2]
        self.original_img = Image.open(f"data/raw/{self.species}/{self.filename}")
        self.ricepr_file = f"data/raw/{self.species}/{self.filename.replace('.jpg', '.ricepr')}"
        self.click_handler = ClickHandler(img_path=self.img_path, ricepr_path=self.ricepr_file)
        self.updated_ricepr = None
        
    def run(self):
        cid = self.click_handler.fig.canvas.mpl_connect('button_press_event', self.click_handler.onclick)
        plt.tight_layout()
        plt.title("Left Click: Add junction -- Right Click: Remove junction\nDon't remove generating junctions")
        plt.show()
        self.updated_ricepr = self.click_handler.update_ricepr()

    def show_update_img(self):
        assert self.updated_ricepr is not None, "Updated ricepr not found! Run InteractiveLabelling().run() first"
        ricepr_manager = riceprManager(PATH=self.updated_ricepr)
        junctions_manager = ricepr_manager.read_ricepr()[0]
        junctions = junctions_manager.return_junctions()
        
        plt.figure(figsize=(10, 9))
        plt.imshow(self.original_img)
        for x, y in junctions:
            plt.scatter(x, y, marker='o', c="yellow", s=35, alpha=0.5)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        