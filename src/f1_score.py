import os
import torch
from ultralytics import YOLOv10
import pandas as pd


def f1_score(img_path, checkpoint, conf, iou_threshold):
    """
    Method: If the IoU between the predicted and true junctions is greater than a threshold, it is a true positive.
    
    Args:
        img_path (str): image path
        checkpoint (str): checkpoint path
        conf (float): confidence cutoff
        iou_threshold (float): IoU threshold
    Returns:
        (tuple): f1, precision, recall
    """
    # ================= #
    #   Retrieve info   #
    # ================= #
    img_name = img_path.split("/")[-1].split(".")[0]
    
    original_folder = "data/raw"
    annotations_folder = "data/annotations"
    
    if os.path.exists(f"{original_folder}/African/{img_name}.jpg"):  # If this is an African rice panicle
        label_txt = f"{annotations_folder}/African/{img_name}_junctions.txt"
    else:  # If this is an Asian rice panicle
        label_txt = f"{annotations_folder}/Asian/{img_name}_junctions.txt"
    
    xywhn_true = []  # We can only get the normalized coordinates of the true junctions
    with open(label_txt, "r") as f:
        lines = f.readlines()
        for line in lines:
            info = line.split(" ")
            x, y, w, h = float(info[1]), float(info[2]), float(info[3]), float(info[4])
            xywhn_true.append([x, y, w, h])
    
    xywhn_true = torch.tensor(xywhn_true)
    num_true = xywhn_true.size()[0]  # Number of true junctions
    
    # ================= #
    #     Inference     # 
    # ================= #
    model = YOLOv10(checkpoint)
    results = model.predict(source=img_path, conf=conf)
    boxes = results[0].boxes
    xywh_pred = boxes.xywh  
    num_pred = xywh_pred.size()[0]  # Number of predicted junctions
    
    # Get xywh as exact coordinates in the image
    height, width = boxes.orig_shape
    xywh_true = xywhn_true * torch.tensor([width, height, width, height])
    
    # ===================================================================================================================== #
    #   Evaluation                                                                                                          #
    #                                                                                                                       #
    #   Explanation:                                                                                                        #
    #       - We create a matrix of IoU between the predicted and true junctions.                                           #
    #       - We count up the TP by 1 if argmax(predicted) is true and argmax(true) is predicted                            #
    #           (because sometimes multiple predictions can have high IoU with the same true box)                           #
    #       - Predicted bounding boxes whose IoU with a matched true box is high but couldn't find a match is counted as FP #
    # ===================================================================================================================== #
    iou_matrix = _compute_iou_matrix(xywh_pred, xywh_true)  # Shape: (num_pred, num_true)
    
    TP = 0
    
    for idx, pred_box_iou in enumerate(iou_matrix):
        pred_box_iou[(pred_box_iou < iou_threshold)] = 0.
        
        while True:
            max_iou_idx = torch.argmax(pred_box_iou)
            
            if pred_box_iou[max_iou_idx] == 0:
                break
            
            if torch.argmax(iou_matrix[:, max_iou_idx]) == idx:
                TP += 1
                iou_matrix[idx, :] = -1
                iou_matrix[:, max_iou_idx] = -1
                break
            else:
                pred_box_iou[max_iou_idx] = 0.
    
    FP = num_pred - TP
    FN = num_true - TP
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1, precision, recall


def _compute_iou(box1, box2):
    """
    Box format: (x, y, w, h)
    
    ⚠️ xywh are exact values in the image
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area


def _compute_iou_matrix(pred_boxes, true_boxes):
    """Row represents predicted boxes, column represents true boxes"""
    num_preds = pred_boxes.size()[0]
    num_trues = true_boxes.size()[0]
    iou_matrix = torch.zeros((num_preds, num_trues))
    
    for i in range(num_preds):
        for j in range(num_trues):
            iou_matrix[i, j] = _compute_iou(pred_boxes[i], true_boxes[j])
    
    return iou_matrix
    

def save_as_excel(history, save_path):
    if os.path.exists(save_path):
        os.remove(save_path)
    df = pd.DataFrame.from_dict(history, orient="index", columns=["f1", "precision", "recall"])
    print(f"==>> Saving {save_path}")
    df.to_excel(save_path)
    
        
if __name__ == "__main__":
    history = dict()
    save_history = True
    split_name = "split2"  # Change this if needed
    save_path = f"logs/{split_name}/val" + "/f1_score.xlsx"
    
    val_folder = "data/splits/val/images"
    for filename in os.listdir(val_folder):
        img_path = f"{val_folder}/{filename}"
        checkpoint = f"checkpoints/{split_name}/run2/best.pt"
        conf = 0.286
        # IoU Threshold should be small because, from experience, iou != 0. means valid prediction.
        # Why small iou means valid prediction? Because some true boxes were not acutely correctly labeled. 
        # One more thing, for small object detection (SOD), small IoU doesn't necessarily mean false prediction [1].
        iou_threshold = 0.001  
        f1, precision, recall = f1_score(img_path=img_path, checkpoint=checkpoint, conf=conf, iou_threshold=iou_threshold)
        print(f"==>> f1: {f1:.2f}, precision: {precision:.2f}, recall: {recall:.2f}")
        history[filename] = (f1, precision, recall)
        break  # Comment out if needed
    
    if save_history:
        save_as_excel(history, save_path)
    
    # [1] YOLOv8-QSD, Wang, 2024, TIM, available at https://ieeexplore.ieee.org/document/10474434.