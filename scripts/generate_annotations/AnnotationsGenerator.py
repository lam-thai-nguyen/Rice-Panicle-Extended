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
        self.bbox_size = 26  # Change this if needed

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
        bbox_size = self.bbox_size

        for x, y in junctions:
            cv2.rectangle(img_copy, pt1=(x - bbox_size//2, y - bbox_size//2), pt2=(x + bbox_size//2, y + bbox_size//2), color=(0, 255, 255), thickness=2)

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
        
        for edge in self.edges:
            x1, y1, x2, y2 = edge
            
            if (x2, y2) in self.junctions.return_terminal():
                
                # ============================================== #
                #     Conditions to avoid small bounding boxes   #
                # ============================================== #
                
                if abs(y1 - y2) <= 10:
                    if y1 < y2:
                        cv2.rectangle(img_copy, pt1=(x1, y1-25), pt2=(x2, y2+25), color=(0, 255, 255), thickness=2)
                    elif y1 >= y2:
                        cv2.rectangle(img_copy, pt1=(x1, y1+25), pt2=(x2, y2-25), color=(0, 255, 255), thickness=2)
                        
                elif abs(y1 - y2) < 25:
                    if y1 < y2:
                        cv2.rectangle(img_copy, pt1=(x1, y1-10), pt2=(x2, y2+10), color=(0, 255, 255), thickness=2)
                    elif y1 >= y2:
                        cv2.rectangle(img_copy, pt1=(x1, y1+10), pt2=(x2, y2-10), color=(0, 255, 255), thickness=2)
                        
                elif abs(x1 - x2) <= 10:
                    if x1 < x2:
                        cv2.rectangle(img_copy, pt1=(x1-25, y1), pt2=(x2+25, y2), color=(0, 255, 255), thickness=2)
                    elif x1 >= x2:
                        cv2.rectangle(img_copy, pt1=(x1+25, y1), pt2=(x2-25, y2), color=(0, 255, 255), thickness=2)
                        
                elif abs(x1 - x2) < 25:
                    if x1 < x2:
                        cv2.rectangle(img_copy, pt1=(x1-10, y1), pt2=(x2+10, y2), color=(0, 255, 255), thickness=2)
                    elif x1 >= x2:
                        cv2.rectangle(img_copy, pt1=(x1+10, y1), pt2=(x2-10, y2), color=(0, 255, 255), thickness=2)
                        
                else:
                    cv2.rectangle(img_copy, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 255), thickness=2)
        
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
        
        for edge in self.edges:
            x1, y1, x2, y2 = edge
            
            if (x2, y2) in self.junctions.return_terminal():
                
                # ============================================== #
                #     Conditions to avoid small bounding boxes   #
                # ============================================== #
                
                if abs(y1 - y2) <= 10:
                    if y1 < y2:
                        xyxy = (x1, y1-25, x2, y2+25)
                        xywh = self._xyxy2xywh(xyxy)
                    elif y1 >= y2:
                        xyxy = (x1, y1+25, x2, y2-25)
                        xywh = self._xyxy2xywh(xyxy)
                        
                elif abs(y1 - y2) < 25:
                    if y1 < y2:
                        xyxy = (x1, y1-10, x2, y2+10)
                        xywh = self._xyxy2xywh(xyxy)
                    elif y1 >= y2:
                        xyxy = (x1, y1+10, x2, y2-10)
                        xywh = self._xyxy2xywh(xyxy)
                        
                elif abs(x1 - x2) <= 10:
                    if x1 < x2:
                        xyxy = (x1-25, y1, x2+25, y2)
                        xywh = self._xyxy2xywh(xyxy)
                    elif x1 >= x2:
                        xyxy = (x1+25, y1, x2-25, y2)
                        xywh = self._xyxy2xywh(xyxy)
                        
                elif abs(x1 - x2) < 25:
                    if x1 < x2:
                        xyxy = (x1-10, y1, x2+10, y2)
                        xywh = self._xyxy2xywh(xyxy)
                    elif x1 >= x2:
                        xyxy = (x1+10, y1, x2-10, y2)
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
                    x, y = int(x) / width, int(y) / height
                    bbox_size = self.bbox_size
                    w, h = bbox_size / width, bbox_size / height
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
    img_path = "data/raw/Asian/10_2_1_1_1_DSC01291.jpg"
    ricepr_path = "data/raw/Asian/10_2_1_1_1_DSC01291.ricepr"
    annotations_generator = AnnotationsGenerator(img_path=img_path, ricepr_path=ricepr_path)
    annotations_generator.draw_junctions(show=True, remove_end_generating=True)
    annotations_generator.draw_grains(show=True)
    annotations_generator.encode_junctions(save_path=".", remove_end_generating=False)
    os.remove("10_2_1_1_1_DSC01291_junctions.txt")
    annotations_generator.encode_grains(save_path=".")
    os.remove("10_2_1_1_1_DSC01291_grains.txt")
    print("All tests passed")


if __name__ == "__main__":
    test()
    