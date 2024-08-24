import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.compute_metrics.F1score import F1score
import yaml


def main(split_name, run_name, mode, conf, iou_threshold):
    f1score = F1score(
        split_name=split_name,
        run_name=run_name,
        mode=mode,
        conf=conf,
        iou_threshold=iou_threshold,
    )
    f1score.compute_save_f1_score()


if __name__ == "__main__":
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    confidence = config["confidence"]
    
    split_name = "split11"
    run_name = ""
    mode = "val"
    conf = confidence[split_name][0] if mode == "train" else confidence[split_name][1]
    iou_threshold = 0.1

    main(split_name, run_name, mode, conf, iou_threshold)
