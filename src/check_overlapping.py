"""
MOTIVATION: 
    - This script creates a plot of overlapping degree at every bounding box size.
    - To do that, it finds the overlapping degree of 560 images at one size.
    - To do that, it finds the overlapping degree of 1 image at one size.
    - To do that, it counts the number of bounding boxes in an image (A), then counts how many boxes exceed the predefined overlapping threshold (B), then compute the overlapping degree as B/A. 
"""


import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
from PIL import Image
from scripts.generate_annotations.AnnotationsGenerator import AnnotationsGenerator
import matplotlib.pyplot as plt


def check_overlapping(split_path: str, percentage: float = 20) -> float:
    """
    Check the overlapping degree (%) over the whole dataset (expectedly 560 images) at ONE SIZE, defined by the data split. 

    Args:
        split_path (str): It is expected to be data/splits/split?/
        percentage (float, optional): It is recommended that the overlapping threshold between stitched images falls within the range of 20% and 50% [1]. Defaults to 20.

    Returns:
        float: The overlapping degree in the range of [0%, 100%]
        
    Ref:
        [1] Shan, Jinhuan, et al. "Unmanned aerial vehicle (UAV)-Based pavement image stitching without occlusion, crack semantic segmentation, and quantification." IEEE Transactions on Intelligent Transportation Systems (2024).
    """
    # Check arguments
    assert 0 <= percentage <= 100, "Percentage must fall within [0, 100]"
    
    # Extract the needed file paths
    train_path = split_path + "/train"
    val_path = split_path + "/val"

    train_img_path = train_path + "/images"
    train_label_path = train_path + "/labels"
    val_img_path = val_path + "/images"
    val_label_path = val_path + "/labels"
    
    # Variables to store important info
    num_boxes_in_total = 0  # Number of boxes in the whole dataset. This one should be consistent in every bounding box sizes.
    num_overlapping_in_total = 0  # Number of boxes that overlap in the whole dataset. This will vary based on bounding box sizes.

    # Check overlapping degree for each image in train/    
    for img_name in os.listdir(train_img_path):
        num_boxes, num_overlapping = check_overlapping_for_single_image(
            img_name=img_name,
            img_dir_path=train_img_path,
            label_dir_path=train_label_path,
            percentage=percentage,
        )

        num_boxes_in_total += num_boxes
        num_overlapping_in_total += num_overlapping

    # Check overlapping degree for each image in val/    
    for img_name in os.listdir(val_img_path):
        num_boxes, num_overlapping = check_overlapping_for_single_image(
            img_name=img_name,
            img_dir_path=val_img_path,
            label_dir_path=val_label_path,
            percentage=percentage,
        )

        num_boxes_in_total += num_boxes
        num_overlapping_in_total += num_overlapping
        
    # Compute the overlapping degree
    overlapping_degree = num_overlapping_in_total / num_boxes_in_total * 100
        
    # Logging out useful info
    print(f"==>> A total of {num_boxes_in_total} objects have been inspected, {num_overlapping_in_total} of which are considered overlapping.")
        
    return overlapping_degree
        

def check_overlapping_for_single_image(img_name, img_dir_path, label_dir_path, percentage) -> tuple:
    """Check overlapping degree (%) for a SINGLE IMAGE, at ONE SIZE"""
    # Variables that store important info
    num_boxes = 0  # Number of boxes in this particular image. This one should be consistent in every bounding box sizes.
    num_overlapping = 0  # Number of boxes that overlap in this particular image. This will vary based on bounding box sizes.
    
    # Get image path
    img_path = img_dir_path + f"/{img_name}"

    # Get image size
    raw_img = Image.open(img_path)
    width, height = raw_img.size

    # Get label path
    label_path = label_dir_path + f"/{img_name[:-len('.jpg')]}.txt"
    with open(label_path, "r") as file:
        boxes = [[float(value) for value in line.split()] for line in file]  # covert every value to float

    # Increment num_boxes with the number of bounding boxes in an image (A)
    num_boxes = len(boxes)

    # Reshape boxes
    for i in range(len(boxes)):
        boxes[i][1] = round(boxes[i][1] * width)
        boxes[i][2] = round(boxes[i][2] * height)
        boxes[i][3] = round(boxes[i][3] * width)
        boxes[i][4] = round(boxes[i][4] * height)

    # Because the width and height are the same for every box, extract that info
    box_width, box_height = boxes[0][3], boxes[0][4]
        
    # Change the way boxes are represented
    boxes = [[box[1], box[2]] for box in boxes]

    # Inspect each box if it overlaps with one other box
    for box in boxes:
        # Get box properties
        x1, y1 = box
        
        # Find the closest box
        x2, y2 = closest_box(x1, y1, boxes)

        # Increment num_overlapping by 1 if it overlaps
        if is_overlapping(x1, y1, x2, y2, box_width, box_height, percentage):
            num_overlapping += 1

    return (num_boxes, num_overlapping)


def closest_box(x1, y1, boxes) -> tuple:
    """Return the closest box."""
    min_dist = float('inf')
    closest_box = None
    
    for x2, y2 in boxes:
        if (x2, y2) == (x1, y1):
            continue
        
        dist = math.dist((x1, y1), (x2, y2))
        if dist < min_dist:
            min_dist = dist
            closest_box = (x2, y2)
    
    return closest_box


def is_overlapping(x_a, y_a, x_b, y_b, box_width, box_height, percentage) -> bool:
    """Returns True if box_a overlaps with box_b"""
    # Create representations for 2 boxes as (x1, y1, x2, y2) with (x1, y1) being the left corner and (x2, y2) being the right corner
    x1_a, y1_a = x_a - box_width // 2, y_a - box_height // 2
    x2_a, y2_a = x_a + box_width // 2, y_a + box_height // 2

    x1_b, y1_b = x_b - box_width // 2, y_b - box_height // 2
    x2_b, y2_b = x_b + box_width // 2, y_b + box_height // 2
    
    # Compute the overlapping region
    x1_overlap, y1_overlap = max(x1_a, x1_b), max(y1_a, y1_b)
    x2_overlap, y2_overlap = min(x2_a, x2_b), min(y2_a, y2_b)
    
    # If x1_overlap >= x2_overlap or y1_overlap >= y2_overlap, the boxes do not overlap
    if x1_overlap >= x2_overlap or y1_overlap >= y2_overlap:
        return False
    
    # Compute the overlapping area, which is the number of overlapping pixels
    area_overlap = (x2_overlap - x1_overlap) * (y2_overlap - y1_overlap)

    # Compute overlap percentage
    overlap_percentage = area_overlap / (box_width * box_height) * 100
    
    return overlap_percentage >= percentage


def show_annotated_images(image_name, bbox_size) -> None:
    """Show the annotate image for debugging purposes."""
    img_path = f"data/raw/African/{image_name}.jpg" if os.path.exists(f"data/raw/African/{image_name}.jpg") else f"data/raw/Asian/{image_name}.jpg"
    ricepr_path = f"data/processed/African/{image_name}.ricepr" if os.path.exists(f"data/processed/African/{image_name}.ricepr") else f"data/processed/Asian/{image_name}.ricepr"

    gen = AnnotationsGenerator(
        img_path=img_path,
        ricepr_path=ricepr_path,
        bbox_size=bbox_size
    )
    
    gen.generate_junctions(show=True, oriented_method=0)
    

if __name__ == "__main__":
    # Check overlapping degree at every bounding box size, which is determined by the split name
    NUM_SPLITS, SPLIT_STEP = 20, 2  # Change this as needed
    MIN_SIZE, MAX_SIZE, STEP = 22, 98, 4 * SPLIT_STEP  # Change the first two as needed
    PERCENTAGE = 20  # Change this if needed

    SPLIT_INDEX = list(range(1, NUM_SPLITS+1, SPLIT_STEP))
    history = list()

    for i in SPLIT_INDEX:
        split_path = f"data/splits/split{i}"
        overlapping_degree = check_overlapping(
            split_path=split_path,
            percentage=PERCENTAGE
        )
        
        history.append(overlapping_degree)
    
    print(f"==>> History: {history}")

    # Plot history
    plt.figure(figsize=(10, 6))
    x_values = list(range(MIN_SIZE, MAX_SIZE+1, STEP))
    plt.scatter(x_values, history, c="red", zorder=2)
    plt.plot(x_values, history, zorder=1)
    plt.xticks(np.arange(MIN_SIZE, MAX_SIZE + 1, STEP))
    plt.xlabel("Bounding box size (pixels)")
    plt.ylabel("Overlapping Degree (%)")
    plt.title("Overlapping degree of bounding boxes based on their size")
    # plt.show()
    
    
    # First derivative
    plt.figure(figsize=(10, 5))
    slopes = np.gradient(history, x_values)
    plt.plot(x_values, slopes, label='First derivative', c='r', marker='o')
    plt.xlabel('Bounding box size')
    plt.legend()
    plt.xticks(np.arange(MIN_SIZE, MAX_SIZE + 1, STEP))

    # Second derivative
    plt.figure(figsize=(10, 5))
    second_slopes = np.gradient(slopes, x_values)
    plt.plot(x_values, second_slopes, label='Second derivative', c='r', marker='o')

    plt.xlabel('Bounding box size')
    plt.legend()
    plt.axhline(y=0, color='black', linestyle=':', linewidth=1)
    plt.xticks(np.arange(MIN_SIZE, MAX_SIZE + 1, STEP))
    plt.show()
    
    # For debugging purposes
    # show_annotated_images(
    #     image_name="10_2_2_1_2_DSC00195",
    #     bbox_size=50
    # )