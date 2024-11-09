# Module Description: compute_metrics

## Structure

```
visualize_predictions
├── F1score.py                     # main class
├── compute_f1_score.py            # util file
└── save_f1_score.py               # util file
```

## Class Description

- This project utilizes the *F<sub>1</sub> score* metric to assess the performance of the YOLO model, rather than mAP.
- **compute_metrics** module helps compute *F<sub>1</sub> score* based on detected bounding boxes (det BBs) and Ground Truth bounding boxes (GT BBs). This module can handle both HBB and OBB.
- The `F1score` class allows us to 
  - `compute_f1_score(save_history: bool)`: Compute *F<sub>1</sub> score*, Precision, Recall and save the metrics as a .xlsx file.
  - `save_f1_score()`: Visualize the .xlsx file and Save it as an image for visual assessment.
  - `compute_save_f1_score()`: Do both mentioned tasks.

## Usage

- Refer to <tt>src/assess_result.py</tt> for usage. This is a reusable file for this purpose.
- Below is the description of this file.

```python
"""
src/assess_result.py -> assess_result()
"""

def assess_result(split_name, mode, conf, iou_threshold=0.1):
    f1score = F1score(
        split_name=split_name,
        mode=mode,
        conf=conf,
        iou_threshold=iou_threshold,
    )
    f1score.compute_save_f1_score()
```

---

- For regular use, just change the specified hyperparameters in <tt>src/visualize_result.py</tt>. 

```python
# HYPERPARAMETERS (USER-DEFINED | CHANGE IF STATIC)
split_name = "split1"  # static
mode = "val"  # static
conf = confidence[split_name][0] if mode == "train" else confidence[split_name][1]  # dynamic
iou_threshold = 0.3  # static | Any value between 0.1 - 0.5 is fine. 

# For SOD, a flexible IOU can be used [1].
# [1] Murrugarra-Llerena et al. (2022). Can we trust bounding box annotations for object detection?. CVPR (pp. 4813-4822).

assess_result(split_name, mode, conf, iou_threshold)
```