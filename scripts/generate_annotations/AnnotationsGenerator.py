import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .riceprManager import riceprManager
from .HorizontalBox import HorizontalBox
from .OrientedBox import OrientedBox
from .SkeletonBasedBox import SkeletonBasedBox


class AnnotationsGenerator:
    def __init__(self, img_path, ricepr_path, bbox_size=None) -> None:
        """
        Create an Annotations Generator for .ricepr files.

        Args:
            img_path (str): original image path
            ricepr_path (str): .ricepr file path
        """
        assert ricepr_path.split("/")[-1].split(".")[-1].lower() == "ricepr", "The given path is not a .ricepr file"
        assert img_path.split("/")[-1].split(".")[0] == ricepr_path.split("/")[-1].split(".")[0], "Unmatched image and .ricepr file"
        
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.ricepr_path = ricepr_path
        self.ricepr_manager = riceprManager(PATH=ricepr_path)
        self.name = self.ricepr_manager.name
        self.species = self.ricepr_manager.species
        self.junctions, self.edges = self.ricepr_manager.read_ricepr()
        self.bbox_size = 26 if bbox_size is None else bbox_size

    def generate_junctions(self, save_path_img=None, show=False, skeleton_based=False, oriented_method=0, save_path_txt=None) -> None:
        """
        Args:
            save_path_img: Specify if you want to save the generated image. Default to None.
            show (bool): Show the generated image? Default to False.
            skeleton_based (bool): Use the junctions from the APSIPA pipeline. Default to False.
            oriented_method = {0, 1, 2}
                0: HBB
                1: OBBv1
                2: OBBv2
            save_path_txt: Specify if you want to ENCODE and save to .txt file. Default to None.
        """
        assert oriented_method in [0, 1, 2], "Invalid oriented_method"
        
        junctions = self.junctions.return_junctions()
        boxes = list()
        
        img_copy = self.img.copy()
        bbox_size = self.bbox_size
        
        if skeleton_based:
            self.junctions.remove_end_generating()
            generating = self.junctions.return_generating()
            primary = self.junctions.return_primary()
            main_axis_junctions = generating + primary
            
            skeleton_based_box = SkeletonBasedBox(
                img_path=self.img_path,
                binary_img_path=f"data/segmentation/{self.species}/{self.name}.jpg"
                # NOTE: This absolute path may cause problem when function is called in different levels away from root
            )
            
            junctions = skeleton_based_box.run(main_axis_junctions)

        if oriented_method:
            oriented_box = OrientedBox(junctions)
            rects = oriented_box.run(width=bbox_size, height=bbox_size, method=oriented_method)  # rects = [(center, (width, height), angle)]
            boxes = rects
            
            for rect in rects:
                obb = cv2.boxPoints(rect)  # 4 corner coords
                obb = np.intp(obb)
                cv2.drawContours(img_copy, [obb], 0, (0, 255, 255), 2)
            
        else:
            horizontal_box = HorizontalBox(junctions)
            rects = horizontal_box.run_junctions(width=bbox_size, height=bbox_size)  # rects = [(pt1, pt2)]
            boxes = junctions
            for pt1, pt2 in rects:
                cv2.rectangle(img_copy, pt1, pt2, (0, 255, 255), 2)
                
        if show:
            self._show(img_copy)
            
        if save_path_img:
            save_path_img = save_path_img + "/" + self.name + "_junctions.jpg"
            print(f"==>> Saving {save_path_img}")
            cv2.imwrite(save_path_img, img_copy)
            
        if save_path_txt:
            print("==>> Encoding junctions")
            self.encode_junctions(boxes, save_path_txt, oriented_method)
            
    def encode_junctions(self, boxes, save_path, method) -> None:
        """
        Args:
            boxes (list)
                HBB: [(x, y), ...]
                OBB: [(x, y, w, h, r), ...]
            method (int): 0: "HBB" or 1: "OBBv1" or 2: "OBBv2"
            save_path (str): file path
        
        Results:
            Horizontal box: (class_index x y w h), normalized between 0 and 1
            Oriented box: (class_index x1 y1 x2 y2 x3 y3 x4 y4), normalized between 0 and 1
            
        Encoding convention references:
            general convention: https://docs.ultralytics.com/datasets/obb/
            point-based OBB convention: https://github.com/orgs/ultralytics/discussions/8462#discussioncomment-9258090
            
        Expected encoding format:
            x1, y1, x4, y4 = topmost corner, leftmost corner
            l1 = x1 - x4
            l2 = y4 - y1
            if (l1 < l2) ==Point-based OBB==>> (x1 y1 x2 y2 x3 y3 x4 y4)  # clockwise from topmost
            if (l2 > l1) ==Point-based OBB==>> (x4 y4 x1 y1 x2 y2 x3 y3)  # clockwise from leftmost
            
            Refer to this image: https://github.com/ultralytics/docs/releases/download/0/obb-format-examples.avif
        """
        save_path = save_path + "/" + self.name + "_junctions.txt"
        print(f"==>> Saving {save_path}")

        # Encoding functions
        def xywhr2xyxyxyxy(boxes: list) -> list:
            xyxyxyxy = list()

            for xywhr in boxes:
                obb = cv2.boxPoints(xywhr)  # 4 corner coords

                # Extract the four corner coordinates of the box
                x1, y1 = obb[np.argmin(obb[:, 1])].tolist()  # topmost in inverted y-axis: lowest y
                x2, y2 = obb[np.argmax(obb[:, 0])].tolist()  # rightmost in inverted y-axis: highest x
                x3, y3 = obb[np.argmax(obb[:, 1])].tolist()  # bottommost in inverted y-axis: highest y
                x4, y4 = obb[np.argmin(obb[:, 0])].tolist()  # leftmost in inverted y-axis: lowest x

                # Compute l1, l2
                l1 = x1 - x4
                l2 = y4 - y1
                
                if l1 <= l2:
                    # Encoding starts from (x1, y1)
                    obb_encoded = tuple([x1, y1, x2, y2, x3, y3, x4, y4])
                else:
                    # Encoding starts from (x4, y4)
                    obb_encoded = tuple([x4, y4, x1, y1, x2, y2, x3, y3])
                
                xyxyxyxy.append(obb_encoded)

            return xyxyxyxy
            
        def normalize(xyxyxyxy: list, img_width, img_height) -> list:
            num_entry = 8
            coords = [float(i) for i in xyxyxyxy]
            normalized_coords = [
                coords[i] / img_width if i % 2 == 0 else coords[i] / img_height for i in range(num_entry)
            ]
            formatted_coords = [f"{coord:.6g}" for coord in normalized_coords]

            return formatted_coords
            
        # Encoding
        height, width, _ = self.img.shape
        
        if method == 0:
            with open(save_path, "w") as f:
                for x, y in boxes:
                    x, y = int(x) / width, int(y) / height
                    bbox_size = self.bbox_size
                    w, h = bbox_size / width, bbox_size / height
                    class_index = 0
                    f.write(f"{class_index} {x:.6g} {y:.6g} {w:.6g} {h:.6g}\n")
                    
        elif method == 1:
            with open(save_path, "w") as f:
                for x1, y1, x2, y2, x3, y3, x4, y4 in xywhr2xyxyxyxy(boxes):
                    x1, y1, x2, y2, x3, y3, x4, y4 = normalize([x1, y1, x2, y2, x3, y3, x4, y4], width, height)
                    class_index = 0
                    f.write(f"{class_index} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
            
        elif method == 2:
            with open(save_path, "w") as f:
                for x1, y1, x2, y2, x3, y3, x4, y4 in xywhr2xyxyxyxy(boxes):
                    x1, y1, x2, y2, x3, y3, x4, y4 = normalize([x1, y1, x2, y2, x3, y3, x4, y4], width, height)
                    class_index = 0
                    f.write(f"{class_index} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")

    # TODO: Implement this function
    def generate_branches(self, level, save_path_img=None, show=False, save_path_txt=None) -> None:
        """
        Generate branches for rice panicles

        Args:
            level (str): ["grains", "primary", "secondary"]
            save_path_img (str): Save path for the generated image. Default to None.
            show (bool): Show the generated image. Default to False.
            save_path_txt (str): Save path for encoded txt. Default to None.
        """
        assert level in ["grains", "primary", "secondary"], "Invalid level"
        img_copy = self.img.copy()
        
        def generate_grains():
            grains = self.ricepr_manager.get_grains()
            return grains

        def generate_primary_branches():
            assert "processed" not in self.ricepr_path, "You have to use the original .ricepr file for this operation"
            primary_branches = self.ricepr_manager.get_primary_branches()
            return primary_branches

        # NOTE: Resolve the logic problem in riceprManager.py first
        def generate_secondary_branches():
            assert "processed" not in self.ricepr_path, "You have to use the original .ricepr file for this operation"
            secondary_branches = self.ricepr_manager.get_secondary_branches()
            return secondary_branches
        
        if level == "grains":
            branches = generate_grains()
        elif level == "primary":
            branches = generate_primary_branches()
        else:
            branches = generate_secondary_branches()
            
        # TODO: Implement HBB for branches
        horizontal_box = HorizontalBox(branches=branches)
        rects = horizontal_box.run_branches()
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if show:
            self._show(img_copy)
            
        if save_path_img:
            save_path_img = save_path_img + "/" + self.name + "_branches.jpg"
            print(f"==>> Saving {save_path_img}")
            cv2.imwrite(save_path_img, img_copy)
            
        if save_path_txt:
            print("==>> Encoding branches")
            self.encode_branches()
            
    # TODO: Implement this function
    def encode_branches(self):
        ...

    def generate_vertex_edge(self, save_path) -> None:
        """draw the vertex and edge from .ricepr file to the image, mainly for debugging purposes"""
        img = self.img.copy()
        terminals = self.junctions.return_terminal()
        primary = self.junctions.return_primary()
        secondary = self.junctions.return_secondary()
        tertiary = self.junctions.return_tertiary()
        generating = self.junctions.return_generating()
        edges = self.edges
        
        for edge in edges:
            x1, y1, x2, y2 = edge
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        for terminal in terminals:
            x, y = terminal
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        for junction in primary:
            x, y = junction
            cv2.circle(img, (x, y), 5, (255, 255, 255), -1)

        for junction in secondary:
            x, y = junction
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

        for junction in tertiary:
            x, y = junction
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        for junction in generating:
            x, y = junction
            cv2.circle(img, (x, y), 5, (255, 255, 0), -1)
        
        if save_path:
            cv2.imwrite(f"{save_path}/{self.name}_vertex_edge.jpg", img)

    def _show(self, img):
        """util function"""
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    def draw_grains(self, save_path=None, show=False):
        # NOTE: DEPRECATED
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
        # NOTE: DEPRECATED
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
                
    def _xyxy2xywh(self, xyxy) -> tuple:
        # NOTE: DEPRECATED
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
    