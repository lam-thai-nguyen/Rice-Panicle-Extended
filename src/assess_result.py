import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.compute_metrics.F1score import F1score


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
    split_name = "split8"
    run_name = ""
    mode = "train"
    conf = 0.233
    iou_threshold = 0.1

    main(split_name, run_name, mode, conf, iou_threshold)
