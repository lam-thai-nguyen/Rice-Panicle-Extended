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
        self.junctions, self.edges = self.ricepr_manager.read_ricepr()
    
    def draw_junctions(self, save_path=None, show=False, remove_end_generating=False):
        if remove_end_generating:
            self.junctions.remove_end_generating()
            assert len(self.junctions.return_generating()) == 1, "Incorrect number of generating junctions"
        
        generating = self.junctions.return_generating()
        primary = self.junctions.return_primary()
        secondary = self.junctions.return_secondary()
        tertiary = self.junctions.return_tertiary()
        quaternary = self.junctions.return_quaternary()
        
        img_copy = self.img.copy()
        
        for x, y in generating:
            cv2.rectangle(img_copy, pt1=(x - 10, y - 10), pt2=(x + 10, y + 10), color=(0, 255, 255), thickness=2)
        for x, y in primary:
            cv2.rectangle(img_copy, pt1=(x - 10, y - 10), pt2=(x + 10, y + 10), color=(0, 255, 255), thickness=2)
        for x, y in secondary:
            cv2.rectangle(img_copy, pt1=(x - 10, y - 10), pt2=(x + 10, y + 10), color=(0, 255, 255), thickness=2)
        for x, y in tertiary:
            cv2.rectangle(img_copy, pt1=(x - 10, y - 10), pt2=(x + 10, y + 10), color=(0, 255, 255), thickness=2)
        for x, y in quaternary:
            cv2.rectangle(img_copy, pt1=(x - 10, y - 10), pt2=(x + 10, y + 10), color=(0, 255, 255), thickness=2)
            
        if show:
            self._show(img_copy)
            
        if save_path:
            cv2.imwrite(save_path, img_copy)
            
    def encode_junctions(self):
        pass
    
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
            cv2.imwrite(save_path, img_copy)
    
    def encode_grains(self):
        pass
    
    def draw_primary_branches(self):
        pass
    
    def encode_primary_branches(self):
        pass
    
    def _show(self, img):
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    

def test():
    img_path = "data/raw/Asian/10_2_1_1_1_DSC01291.jpg"
    ricepr_path = "data/raw/Asian/10_2_1_1_1_DSC01291.ricepr"
    annotations_generator = AnnotationsGenerator(img_path=img_path, ricepr_path=ricepr_path)
    annotations_generator.draw_junctions(show=True)
    annotations_generator.draw_junctions(show=True, remove_end_generating=True)
    annotations_generator.draw_grains(show=True)
    print("All tests passed")
    

if __name__ == "__main__":
    test()
    