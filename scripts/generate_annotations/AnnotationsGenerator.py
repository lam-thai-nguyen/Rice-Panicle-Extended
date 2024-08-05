import os
import cv2
import matplotlib.pyplot as plt
from .riceprManager import riceprManager


class AnnotationsGenerator:
    def __init__(self, img_path, ricepr_path):
        """
        Create an Annotations Generator for .ricepr files.

        Args:
            img_path (str): original image path
            ricepr_path (str): .ricepr file path
        """
        assert type(img_path) == str, "Invalid type" 
        assert type(ricepr_path) == str, "Invalid type" 
        assert img_path.split("/")[-1].split(".")[-1].lower() == "jpg", "The given path is not an image"
        assert ricepr_path.split("/")[-1].split(".")[-1].lower() == "ricepr", "The given path is not a .ricepr file"
        assert img_path.split("/")[-1].split(".")[0] == ricepr_path.split("/")[-1].split(".")[0], "Unmatched image and .ricepr file"
        
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.ricepr_path = ricepr_path
        self.ricepr_manager = riceprManager(PATH=ricepr_path)
        self.name = self.ricepr_manager.name
        self.species = self.ricepr_manager.species
        self.junctions, self.edges = self.ricepr_manager.read_ricepr()
        self.factor = 1.
        
    def size(self):
        """Returns the size of the image as (H, W, C) or (rows, cols, channels)"""
        return self.img.shape
    
    def upscale(self, factor=2.):
        """Upscale the image by the given factor"""
        self.factor = factor
        self.img = cv2.resize(self.img, None, fx=factor, fy=factor)
    
    def draw_junctions(self, save_path=None, show=False, remove_end_generating=False):
        if remove_end_generating and len(self.junctions.return_generating()) == 2:
            self.junctions.remove_end_generating()
        
        generating = self.junctions.return_generating()
        primary = self.junctions.return_primary()
        secondary = self.junctions.return_secondary()
        tertiary = self.junctions.return_tertiary()
        quaternary = self.junctions.return_quaternary()
        junctions = generating + primary + secondary + tertiary + quaternary
        
        img_copy = self.img.copy()
        
        for x, y in junctions:
            # Upscaling
            x, y = int(x * self.factor), int(y * self.factor)
            offset = int(13 * self.factor)
            thickness = 1 + int(self.factor)
            
            # Draw
            cv2.rectangle(img_copy, pt1=(x - offset, y - offset), pt2=(x + offset, y + offset), color=(0, 255, 255), thickness=thickness)
            
        if show:
            self._show(img_copy)
            
        if save_path:
            save_path = save_path + "/" + self.name + "_junctions.jpg"
            print(f"==>> Saving {save_path}")
            cv2.imwrite(save_path, img_copy)
            
    def encode_junctions(self, save_path=None, remove_end_generating=False):
        """
        Encode the junction bounding boxes as (class_label, x, y, w, h), all relative to the whole image
        """
        if remove_end_generating and len(self.junctions.return_generating()) == 2:
            self.junctions.remove_end_generating()

        generating = self.junctions.return_generating()
        primary = self.junctions.return_primary()
        secondary = self.junctions.return_secondary()
        tertiary = self.junctions.return_tertiary()
        quaternary = self.junctions.return_quaternary()
        
        junctions = generating + primary + secondary + tertiary + quaternary
        
        if save_path:
            save_path = save_path + "/" + self.name + "_junctions.txt"
            print(f"==>> Saving {save_path}")
            self._encode_box(junctions, save_path, mode="junctions")
            
    def draw_grains(self, save_path=None, show=False):
        img_copy = self.img.copy()
        
        # Upscaling
        terminal = self.junctions.return_terminal()
        terminal = [(int(x * self.factor), int(y * self.factor)) for x, y in terminal]
        thickness = 1 + int(self.factor)
        
        for edge in self.edges:
            x1, y1, x2, y2 = edge
            x1, y1, x2, y2 = int(x1 * self.factor), int(y1 * self.factor), int(x2 * self.factor), int(y2 * self.factor)
            
            if (x2, y2) in terminal:
                
                # ============================================== #
                #     Conditions to avoid small bounding boxes   #
                # ============================================== #
                
                if abs(y1 - y2) <= 10 * self.factor:
                    offset = int(25 * self.factor)
                    if y1 < y2:
                        cv2.rectangle(img_copy, pt1=(x1, y1-offset), pt2=(x2, y2+offset), color=(0, 255, 255), thickness=thickness)
                    elif y1 >= y2:
                        cv2.rectangle(img_copy, pt1=(x1, y1+offset), pt2=(x2, y2-offset), color=(0, 255, 255), thickness=thickness)
                        
                elif abs(y1 - y2) < 25 * self.factor:
                    offset = int(10 * self.factor)
                    if y1 < y2:
                        cv2.rectangle(img_copy, pt1=(x1, y1-offset), pt2=(x2, y2+offset), color=(0, 255, 255), thickness=thickness)
                    elif y1 >= y2:
                        cv2.rectangle(img_copy, pt1=(x1, y1+offset), pt2=(x2, y2-offset), color=(0, 255, 255), thickness=thickness)
                        
                elif abs(x1 - x2) <= 10 * self.factor:
                    offset = int(25 * self.factor)
                    if x1 < x2:
                        cv2.rectangle(img_copy, pt1=(x1-offset, y1), pt2=(x2+offset, y2), color=(0, 255, 255), thickness=thickness)
                    elif x1 >= x2:
                        cv2.rectangle(img_copy, pt1=(x1+offset, y1), pt2=(x2-offset, y2), color=(0, 255, 255), thickness=thickness)
                        
                elif abs(x1 - x2) < 25 * self.factor:
                    offset = int(10 * self.factor)
                    if x1 < x2:
                        cv2.rectangle(img_copy, pt1=(x1-offset, y1), pt2=(x2+offset, y2), color=(0, 255, 255), thickness=thickness)
                    elif x1 >= x2:
                        cv2.rectangle(img_copy, pt1=(x1+offset, y1), pt2=(x2-offset, y2), color=(0, 255, 255), thickness=thickness)
                        
                else:
                    cv2.rectangle(img_copy, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 255), thickness=thickness)
        
        if show:
            self._show(img_copy)
            
        if save_path:
            save_path = save_path + "/" + self.name + "_grains.jpg"
            print(f"==>> Saving {save_path}")
            cv2.imwrite(save_path, img_copy)
    
    def encode_grains(self, save_path=None):
        """
        Encode the grain bounding boxes as (class_label, x, y, w, h), all relative to the whole image
        """
        grains = list()
        
        # Upscaling
        terminal = self.junctions.return_terminal()
        terminal = [(int(x * self.factor), int(y * self.factor)) for x, y in terminal]
        
        for edge in self.edges:
            x1, y1, x2, y2 = edge
            x1, y1, x2, y2 = int(x1 * self.factor), int(y1 * self.factor), int(x2 * self.factor), int(y2 * self.factor)
            
            if (x2, y2) in terminal:
                
                # ============================================== #
                #     Conditions to avoid small bounding boxes   #
                # ============================================== #
                
                if abs(y1 - y2) <= 10 * self.factor:
                    offset = int(25 * self.factor)
                    if y1 < y2:
                        xyxy = (x1, y1-offset, x2, y2+offset)
                        xywh = self._xyxy2xywh(xyxy)
                    elif y1 >= y2:
                        xyxy = (x1, y1+offset, x2, y2-offset)
                        xywh = self._xyxy2xywh(xyxy)
                        
                elif abs(y1 - y2) < 25 * self.factor:
                    offset = int(10 * self.factor)
                    if y1 < y2:
                        xyxy = (x1, y1-offset, x2, y2+offset)
                        xywh = self._xyxy2xywh(xyxy)
                    elif y1 >= y2:
                        xyxy = (x1, y1+offset, x2, y2-offset)
                        xywh = self._xyxy2xywh(xyxy)
                        
                elif abs(x1 - x2) <= 10 * self.factor:
                    offset = int(25 * self.factor)
                    if x1 < x2:
                        xyxy = (x1-offset, y1, x2+offset, y2)
                        xywh = self._xyxy2xywh(xyxy)
                    elif x1 >= x2:
                        xyxy = (x1+offset, y1, x2-offset, y2)
                        xywh = self._xyxy2xywh(xyxy)
                        
                elif abs(x1 - x2) < 25 * self.factor:
                    offset = int(10 * self.factor)
                    if x1 < x2:
                        xyxy = (x1-offset, y1, x2+offset, y2)
                        xywh = self._xyxy2xywh(xyxy)
                    elif x1 >= x2:
                        xyxy = (x1+offset, y1, x2-offset, y2)
                        xywh = self._xyxy2xywh(xyxy)
                        
                else:
                    xyxy = (x1, y1, x2, y2)
                    xywh = self._xyxy2xywh(xyxy)
                    
                grains.append(xywh)
        
        if save_path:
            save_path = save_path + "/" + self.name + "_grains.txt"
            print(f"==>> Saving {save_path}")
            self._encode_box(grains, save_path, mode="grains")
    
    def draw_junctions_grains(self):
        pass
    
    def encode_junctions_grains(self):
        pass
    
    def draw_primary_branches(self):
        pass
    
    def encode_primary_branches(self):
        pass
    
    def _show(self, img):
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    def _encode_box(self, boxes, save_path, mode) -> None:
        """
        Args:
            boxes (list): list of all boxes
            save_path (str): file path
            mode (str): "junctions" or "grains"
        """
        assert mode in ["junctions", "grains"], "Invalid mode"
        
        height, width, _ = self.img.shape
        
        if mode == "junctions":
            with open(save_path, "w") as f:
                for x, y in boxes:
                    # Upscaling
                    x, y = x * self.factor, y * self.factor
                    
                    # Encoding
                    x, y = int(x) / width, int(y) / height
                    w, h = 26 * self.factor / width, 26 * self.factor / height
                    class_label = 0
                    f.write(f"{class_label} {x} {y} {w} {h}\n")
                    
        elif mode == "grains":
            with open(save_path, "w") as f:
                for x, y, w, h in boxes:
                    x, y = x / width, y / height
                    w, h = w / width, h / height
                    class_label = 0
                    f.write(f"{class_label} {x} {y} {w} {h}\n")
                
    def _xyxy2xywh(self, xyxy) -> tuple:
        """
        Turns a bounding box in (x1, y1, x2, y2) format into (x, y, w, h)

        Args:
            xyxy (tuple): (x1, y1, x2, y2)
            
        Returns:
            tuple: (x, y, w, h)
        """
        x1, y1, x2, y2 = xyxy
        
        # Compute the minimum and maximum coordinates
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)
        
        # Compute width and height
        width = x_max - x_min
        height = y_max - y_min
        
        # Compute center coordinates
        x_center = x_min + width / 2
        y_center = y_min + height / 2
    
        xywh = (x_center, y_center, width, height)
        
        return xywh

def test():
    img_path = "data/raw/Asian/11_2_1_1_2_DSC01180.jpg"
    ricepr_path = "data/raw/Asian/11_2_1_1_2_DSC01180.ricepr"
    annotations_generator = AnnotationsGenerator(img_path=img_path, ricepr_path=ricepr_path)
    annotations_generator.upscale()
    annotations_generator.draw_junctions(show=True, remove_end_generating=True)
    annotations_generator.draw_grains(show=True)
    annotations_generator.encode_junctions(save_path=".", remove_end_generating=True)
    os.remove("11_2_1_1_2_DSC01180_junctions.txt")
    annotations_generator.encode_grains(save_path=".")
    os.remove("11_2_1_1_2_DSC01180_grains.txt")
    print("All tests passed")
    

if __name__ == "__main__":
    test()
    