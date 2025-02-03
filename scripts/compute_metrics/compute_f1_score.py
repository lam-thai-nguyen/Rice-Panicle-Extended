import os
import torch
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Polygon


def compute_f1_score(img_path, label_path, checkpoint, conf, iou_threshold) -> tuple[float, float, float]:
    """
    Method: If the IoU between the predicted and true junctions is greater than a threshold, it is a true positive.
    
    Args:
        img_path (str): image path
        label_path (str): .txt file path
        checkpoint (str): checkpoint path
        conf (float): confidence cutoff
        iou_threshold (float): IoU threshold
    Returns:
        (tuple): f1, precision, recall
    """
    # Load model and get task
    model = YOLO(checkpoint)
    
    if model.task == "detect":
        flag = "HBB"
    elif model.task == "obb":
        flag = "OBB"
        # NOTE: FYI, result.obb.xywhr has rotation angle in radian, within [-pi, pi]. 
        # Answered by Glenn Jocher at https://github.com/ultralytics/ultralytics/issues/15677#issuecomment-2295746033.
    else:
        raise Exception("Unexpected YOLO model's task.")
    
    # Functions with behaviors based on flags
    def get_GT(flag, label_path) -> tuple[torch.Tensor, int]:
        width, height = Image.open(img_path).size
        
        if flag == "HBB":
            xywhn_GT = list()
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    info = line.split(" ")
                    x, y, w, h = float(info[1]), float(info[2]), float(info[3]), float(info[4])  # x, y is center point
                    xywhn_GT.append([x, y, w, h])
            
            xywhn_GT = torch.tensor(xywhn_GT)  # Turn into a 2D Tensor of shape (num_GT, 4)
            num_GT = xywhn_GT.size()[0]  # Number of GT junctions
            xywh_GT = xywhn_GT * torch.tensor([width, height, width, height])
            GT = tuple([xywh_GT, num_GT])
        else:
            xyxyxyxyn_GT = list()
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    info = line.split(" ")
                    x1, y1 = float(info[1]), float(info[2])
                    x2, y2 = float(info[3]), float(info[4])
                    x3, y3 = float(info[5]), float(info[6])
                    x4, y4 = float(info[7]), float(info[8])
                    xyxyxyxyn_GT.append([x1, y1, x2, y2, x3, y3, x4, y4])
            
            xyxyxyxyn_GT = torch.tensor(xyxyxyxyn_GT)
            num_GT = xyxyxyxyn_GT.size()[0]  # Number of GT junctions
            xyxyxyxy_GT = xyxyxyxyn_GT * torch.tensor([width, height, width, height, width, height, width, height])
            GT = tuple([xyxyxyxy_GT, num_GT])
            
        return GT
            
    def get_det(flag) -> tuple[torch.Tensor, int]:
        results = model.predict(source=img_path, conf=conf)
        if flag == "HBB":
            boxes = results[0].boxes
            xywh_det = boxes.xywh  # x, y is center point
            num_det = xywh_det.size()[0]  # Number of detected junctions
            det = tuple([xywh_det, num_det])
        else:
            obb = results[0].obb
            xyxyxyxy_det = obb.xyxyxyxy
            xyxyxyxy_det = torch.flatten(xyxyxyxy_det, start_dim=1, end_dim=-1)  # (num_det, num_obb, num_coord) -> (num_det, num_obb*num_coord)
            num_det = xyxyxyxy_det.size()[0]  # Number of detected junctions
            det = tuple([xyxyxyxy_det, num_det])
            
        return det
        
    def get_IOU(flag, box1, box2):
        if flag == "HBB":
            """Box format: (x_center, y_center, w, h) -> absolute values, not normalized"""
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2

            # Convert (x_center, y_center, w, h) to (x_min, y_min, x_max, y_max)
            x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
            x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
            x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
            x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

            # Get intersection corners
            xi1 = max(x1_min, x2_min)
            yi1 = max(y1_min, y2_min)
            xi2 = min(x1_max, x2_max)
            yi2 = min(y1_max, y2_max)
            
            # Compute IOU
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - inter_area
            IOU = inter_area / union_area
        else:
            """Box format: (x1, y1, x2, y2, x3, y3, x4, y4) -> absolute values, not normalized"""
            # Create Polygon()
            poly1 = Polygon([(box1[0], box1[1]), (box1[2], box1[3]), (box1[4], box1[5]), (box1[6], box1[7])])
            poly2 = Polygon([(box2[0], box2[1]), (box2[2], box2[3]), (box2[4], box2[5]), (box2[6], box2[7])])
            
            # Compute IOU
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            IOU = inter_area / union_area
            
        return IOU
        
    def get_IOU_matrix(flag, det, GT):
        """Row represents detected boxes, column represents GT boxes"""
        num_det = det.size()[0]
        num_GT = GT.size()[0]
        iou_matrix = torch.zeros((num_det, num_GT))
        
        for i in range(num_det):
            for j in range(num_GT):
                iou_matrix[i, j] = get_IOU(flag, det[i], GT[j])
        
        return iou_matrix

    # Retrieving ground truth (GT) and detection (det)
    GT, num_GT = get_GT(flag, label_path)
    det, num_det = get_det(flag)
    
    # ========================================================================================================================= #
    #   Computing F1, Pr, Rc                                                                                                    #
    #                                                                                                                           #
    #   Motivation (Idea explanation):                                                                                          #
    #       - We create a matrix of IoU between the *detected* and *true* junctions.                                            #
    #       - We count up the TP by 1 if argmax(*detected*) is *true* and argmax(*true*) is *detected*                          #
    #           (because sometimes multiple detections can have high IoU with the same *true* box)                              #
    #       - *Detected* bounding boxes whose IoU with a matched *true* box is high but couldn't find a match is counted as FP  #
    # ========================================================================================================================= #
    
    iou_matrix = get_IOU_matrix(flag, det, GT)  # Shape: (num_det, num_GT)
    
    TP = 0
    
    for idx, det_iou in enumerate(iou_matrix):
        det_iou[(det_iou < iou_threshold)] = 0.  # In a row, every IOU below the threshold is zeroed.
        
        while True:
            max_iou_idx = torch.argmax(det_iou)  # Get the column index where IOU is max
            
            if det_iou[max_iou_idx] == 0:  # If the max IOU is 0, skip this detection
                break
            
            if torch.argmax(iou_matrix[:, max_iou_idx]) == idx:  # If the current column (max_iou_idx) has the current row (idx) as the max IOU
                TP += 1  # Count this IOU as 1 True Positive
                iou_matrix[idx, :] = -1  # Eliminate the current row (detection) from further examination
                iou_matrix[:, max_iou_idx] = -1  # Eliminate the found column (GT) from further examination
                break
            else:
                det_iou[max_iou_idx] = 0.
    
    FP = num_det - TP
    FN = num_GT - TP
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1, precision, recall


def save_as_excel(history, save_path):
    if os.path.exists(save_path):
        os.remove(save_path)
    df = pd.DataFrame.from_dict(history, orient="index", columns=["f1", "precision", "recall"])
    print(f"==>> Saving {save_path}")
    df.to_excel(save_path)
    
        
if __name__ == "__main__":
    history = dict()
    save_history = True  # Change this if needed
    mode = "val"  # Change this if needed [train or val]
    split_name = "split25"  # Change this if needed
    save_path = f"./f1_score.xlsx"
    
    images_folder = f"data/splits/{split_name}/{mode}/images"
    labels_folder = f"data/splits/{split_name}/{mode}/labels"
    
    with tqdm(os.listdir(images_folder), desc="Evaluating") as pbar:
        for filename in pbar:
            # Configuration
            img_path = f"{images_folder}/{filename}"
            label_path = f"{labels_folder}/{filename[:-4]}.txt"
            checkpoint = f"checkpoints/{split_name}/best.pt"
            
            conf = 0.376  # Change this if needed
            # ============================================================================================================= #
            # IoU Threshold should be small because, from experience, iou != 0. means valid prediction.                     #
            # Why small iou means valid prediction? Because some true boxes were not acutely correctly labeled.             #
            # One more thing, for small object detection (SOD), small IoU doesn't necessarily mean false prediction [1].    #
            # ============================================================================================================= #
            iou_threshold = 0.1  # Change this if needed

            # Compute metrics
            f1, precision, recall = compute_f1_score(
                img_path=img_path, 
                label_path=label_path,
                checkpoint=checkpoint, 
                conf=conf, 
                iou_threshold=iou_threshold
            )

            # Update progression bar
            pbar.set_postfix({"F1": f"{f1:.2f}", "P": f"{precision:.2f}", "R": f"{recall:.2f}"})
            
            # Update history
            history[filename] = (f1, precision, recall)

            # break  # Comment out if needed
    
    if save_history:
        save_as_excel(history, save_path)
    
    # [1] YOLOv8-QSD, Wang, 2024, TIM, available at https://ieeexplore.ieee.org/document/10474434.