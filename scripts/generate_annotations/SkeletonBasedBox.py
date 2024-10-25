"""
The idea of skeleton-based bounding box is automatically generating bounding box annotation for junction detection.
That is, we don't have to manually annotate each image.
To get a skeleton-based box, following steps are required:
    (1) Original image (?, ?)                   ---Segmentation-->                  Mask (512, 512)
    (2) Mask (512, 512)                         ---Zhang Suen Thinning-->           Skeleton (512, 512)
    (3) Skeleton (512, 512)                     ---Crossing Number-->               junctions: list (512, 512)
    (4) junctions (512, 512)                    ---Conversion-->                    junctions (?, ?)
    (5) `oriented_box = OrientedBox(junctions)` ---Refer to ./OrientedBox.py-->     skeleton-based box
    
    # NOTE: Additional implementation details can be found below, including (1) resizing original junctions (2) merging high order junctions
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN

SEGMENTATION_MASK_SIZE = (512, 512)


class SkeletonBasedBox:
    def __init__(self, img_path, binary_img_path) -> None:
        self.img = cv2.imread(img_path)
        self.orig_size = self.img.shape[:2][::-1]  # (width, height)
        self.binary_img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)
        
    def run(self, main_axis_junctions) -> list:
        """
        Main axis junctions do not include end generating point, for convenience purpose.

        Example input for `main_axis_junctions`: `generating = [sorted(generating, key=lambda x: x[0])[-1]]  # Remove end generating point`
        
        Note: `main_axis_junctions` comes as (x, y), while `crossing_number()` returns (y, x). 
        """
        src_size = self.orig_size  # (width, height)
        
        skeleton_img = self.zhang_suen(self.binary_img)
        intersection_pts = self.crossing_number(skeleton_img)  # (y, x)

        main_axis_junctions_resized = self.resize_junctions(main_axis_junctions, src_size, SEGMENTATION_MASK_SIZE)
        skeleton_main_axis_img = self.get_main_axis_skeleton(skeleton_img, main_axis_junctions_resized)
        main_axis_intersection_pts = self.crossing_number(skeleton_main_axis_img)  # (y, x)
        
        high_order_intersection_pts = [pts for pts in intersection_pts if pts not in main_axis_intersection_pts]
        high_order_intersection_pts_merged = self.merge_high_order_junctions(high_order_intersection_pts)
        
        junctions = main_axis_intersection_pts + high_order_intersection_pts_merged  # (y, x)
        junctions = [[point[1], point[0]] for point in junctions]  # Convert to (x, y)
        junctions_resized = self.resize_junctions(junctions, SEGMENTATION_MASK_SIZE, src_size)  # Convert back to original size

        return junctions_resized
         
    def zhang_suen(self, binary_img):
        # Thresholding
        _, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)

        # Extracting skeletons
        skeleton_img = skeletonize(binary_img, method="zhang").astype(np.uint8) * 255
        return skeleton_img
    
    # NOTE: Output pixel coordinate is (y, x) 
    def crossing_number(self, skeleton_img) -> list:
        img = np.copy(skeleton_img)
    
        # White px intensity 255 -> 1
        img[img == 255] = 1
        white_px = np.argwhere(img > 0)
        intersection_pts = list()

        # Crossing number
        for row, col in white_px:
            row, col = int(row), int(col)

            try:
                P1 = img[row, col + 1].astype("i")
                P2 = img[row - 1, col + 1].astype("i")
                P3 = img[row - 1, col].astype("i")
                P4 = img[row - 1, col - 1].astype("i")
                P5 = img[row, col - 1].astype("i")
                P6 = img[row + 1, col - 1].astype("i")
                P7 = img[row + 1, col].astype("i")
                P8 = img[row + 1, col + 1].astype("i")
            except:
                continue

            crossing_number = abs(P2 - P1) + abs(P3 - P2) + abs(P4 - P3) + abs(P5 - P4) + abs(P6 - P5) + abs(P7 - P6) + abs(P8 - P7) + abs(P1 - P8)
            crossing_number //= 2
            if crossing_number == 3 or crossing_number == 4:
                intersection_pts.append((row, col))

        return intersection_pts
    
    def resize_junctions(self, junctions, src_size, dst_size) -> list:
        src_width, src_height = src_size
        dst_width, dst_height = dst_size
        
        junctions_resized = list()
        for (src_x, src_y) in junctions:
            dst_x = round((src_x / src_width) * dst_width)
            dst_y = round((src_y / src_height) * dst_height)
            junctions_resized.append((dst_x, dst_y))

        return junctions_resized
    
    def get_main_axis_skeleton(self, skeleton_img, main_axis_junctions_resized):
        skeleton_main_axis_img = np.copy(skeleton_img)
        x_min, x_max = min(point[0] for point in main_axis_junctions_resized), max(point[0] for point in main_axis_junctions_resized)
        y_min, y_max = min(point[1] for point in main_axis_junctions_resized), max(point[1] for point in main_axis_junctions_resized)
        
        skeleton_main_axis_img[:y_min-4, :] = 0
        skeleton_main_axis_img[y_max+5:, :] = 0
        skeleton_main_axis_img[:, :x_min-4] = 0
        skeleton_main_axis_img[:, x_max+5:] = 0
        
        return skeleton_main_axis_img
    
    def merge_high_order_junctions(self, high_order_intersection_pts) -> list:
        high_order_intersection_pts_merged = high_order_intersection_pts.copy()
        
        high_order_intersection_pts_np = np.array(high_order_intersection_pts)
        db = DBSCAN(eps=7, min_samples=2).fit(high_order_intersection_pts_np)
        labels = db.labels_
        
        # Merging
        for label in np.unique(labels):
            if label != -1:
                pts = high_order_intersection_pts_np[labels == label]
                for point in pts:
                    high_order_intersection_pts_merged.remove(tuple(point))

                x, y = np.mean(pts, axis=0).astype("i")
                high_order_intersection_pts_merged.append((x, y))
                
        return high_order_intersection_pts_merged
    