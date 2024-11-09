import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.compute_metrics.F1score import F1score
import yaml


def assess_result(split_name, mode, conf, iou_threshold=0.1):
    f1score = F1score(
        split_name=split_name,
        mode=mode,
        conf=conf,
        iou_threshold=iou_threshold,
    )
    f1score.compute_save_f1_score()


if __name__ == "__main__":
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    confidence = config["confidence"]
    
    # HYPERPARAMETERS (USER-DEFINED | CHANGE IF STATIC)
    split_name = "split1"  # static
    mode = "val"  # static
    conf = confidence[split_name][0] if mode == "train" else confidence[split_name][1]  # dynamic
    iou_threshold = 0.3  # static | Any value between 0.1 - 0.5 is fine. 
    
    # For SOD, a flexible IOU can be used [1].
    # [1] Murrugarra-Llerena et al. (2022). Can we trust bounding box annotations for object detection?. CVPR (pp. 4813-4822).

    assess_result(split_name, mode, conf, iou_threshold)
