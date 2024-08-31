import os
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import Tk, messagebox
from ..generate_annotations.riceprManager import riceprManager
from .ClickHandler import ClickHandler


class InteractiveLabelling:
    def __init__(self, img_path, save_path) -> None:
        """
        img_path: path to image to display (i.e., non-processed ground truth)
        save_path: path to updated .ricepr file's parent, names and species are generated automatically
        """
        self.img_path = img_path
        self.filename = self.img_path.split("/")[-1]
        self.species = self.img_path.split("/")[-2]
        self.save_path = save_path
        
        self.raw_dir = "data/raw"
        self.orig_img = Image.open(f"{self.raw_dir}/{self.species}/{self.filename}")
        self.orig_ricepr = f"{self.raw_dir}/{self.species}/{self.filename.replace('.jpg', '.ricepr')}"
        
        self.click_handler = ClickHandler(img_path=self.img_path, orig_ricepr=self.orig_ricepr, save_path=self.save_path)
        self.updated_ricepr = None  # Generated updated .ricepr file, with species and name
        
    def run(self) -> None:
        cid_click = self.click_handler.fig.canvas.mpl_connect('button_press_event', self.click_handler.onclick)
        
        # Uncomment this to ask users yes/no after the interactive plot
        # cid_close = self.click_handler.fig.canvas.mpl_connect('close_event', self.click_handler.on_close)

        plt.title("Left Click: Add junction -- Right Click: Remove junction\nDon't remove generating junctions")
        plt.tight_layout()
        plt.show()
        self.updated_ricepr = self.click_handler.update_ricepr()

    def show_update_img(self):
        assert self.updated_ricepr is not None, "Updated ricepr not found! Run InteractiveLabelling().run() first"
        print(f"==>> InteractiveLabelling - Showing updated image")

        # Read and extract updated junctions
        ricepr_manager = riceprManager(PATH=self.updated_ricepr)
        junctions = ricepr_manager.read_ricepr()[0].return_junctions()
        
        # Plot
        plt.figure(figsize=(10, 9))
        plt.imshow(self.orig_img)
        for x, y in junctions:
            plt.scatter(x, y, marker='o', c="yellow", s=35, alpha=0.5)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        # Uncomment this to unable asking users yes/no after the updated plot
        self.on_close()
        
    def on_close(self):
        root = Tk()
        root.withdraw()

        if messagebox.askyesno("Save Edits", "Do you want to save your edits before closing?\n\nImportant: You can't change the .ricepr file afterwards.\nHint: Change the image name back to original."):
            self.save_edits()
        root.destroy()
        
    def save_edits(self):
        """
        Function to save edits (to be implemented as per requirements).
        """
        print("==>> InteractiveLabelling - Saved edits")
        # Rename non-processed ground truth image - Mark as [done]
        src = self.img_path
        dst = f"{self.save_path}/{self.species}/[done] {self.filename}"
        os.rename(src, dst)