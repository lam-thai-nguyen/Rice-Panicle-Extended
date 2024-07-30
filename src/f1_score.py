import os
import torch
from ultralytics import YOLOv10


def f1score(img_path, checkpoint, conf):
    """
    Method: If the IoU between the predicted and true junctions is greater than 0.5, it is a true positive.
    
    Args:
        img_path (str): image path
        checkpoint (str): checkpoint path
        conf (float): confidence cutoff
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
    
    # ================= #
    #     Evaluation    #
    # ================= #
    iou_matrix = _compute_iou_matrix(xywh_pred, xywh_true)  # Shape: (num_pred, num_true)
    
    max_iou_per_pred = torch.max(iou_matrix, dim=1).values  # Shape: (num_pred,)
    iou_threshold = 0.5
    
    TP = torch.sum((max_iou_per_pred > iou_threshold))
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
    
    
if __name__ == "__main__":
    val_folder = "data/splits/val/images"
    for filename in os.listdir(val_folder):
        img_path = f"{val_folder}/{filename}"
        checkpoint = "checkpoints/best.pt"
        conf = 0.289
        f1, precision, recall = f1score(img_path=img_path, checkpoint=checkpoint, conf=conf)
        print(f"==>> f1: {f1:.2f}, precision: {precision:.2f}, recall: {recall:.2f}")
        break  # Comment out if needed
    