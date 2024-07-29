import os
import cv2
import torch
from ultralytics import YOLOv10
from matplotlib import pyplot as plt


def visualize_result(img_path, conf, mode, show=False, save_path=None):
    """
    Args:
        img_path (str): image path
        conf (float): confidence cutoff
        mode (str): "single" or "side" or "overlay"
        show (bool, optional): Defaults to False.
        save_path (str, optional): Parent dir (filename will be generated). Defaults to None.
    """
    assert mode in ["single", "side", "overlay"], "Invalid mode"
    
    # ================= #
    #   Retrieve info   #
    # ================= #
    img_name = img_path.split("/")[-1].split(".")[0]
    
    original_folder = "data/raw"
    annotations_folder = "data/annotations"
    
    # Get annotation image and annotation txt
    if os.path.exists(f"{original_folder}/African/{img_name}.jpg"):  # If this is an African rice panicle
        label_img = cv2.imread(f"{annotations_folder}/African/{img_name}_junctions.jpg")
        label_txt = f"{annotations_folder}/African/{img_name}_junctions.txt"
    else:  # If this is an Asian rice panicle
        label_img = cv2.imread(f"{annotations_folder}/Asian/{img_name}_junctions.jpg")
        label_txt = f"{annotations_folder}/Asian/{img_name}_junctions.txt"
    
    label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
    num_true = len(open(label_txt, "r").readlines())
    
    # ================= #
    #     Inference     # 
    # ================= #
    checkpoint = "checkpoints/best.pt"
    model = YOLOv10(checkpoint)
    
    results = model.predict(source=img_path, conf=conf)
    result = results[0]
    num_pred = len(result.boxes)
    
    pred_img = result.plot(conf=False, labels=False)
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    
    # ================= #
    #     Visualize     #
    # ================= #
    if mode == "single":
        plt.figure(figsize=(8, 8))
        plt.imshow(pred_img)
        plt.axis("off")
        plt.tight_layout()
        plt.title(f"{num_pred} junctions")
        if show:
            plt.show()
        
    elif mode == "side":
        _plot_side(pred_img, label_img, show, num_pred, num_true)
    
    elif mode == "overlay":
        _plot_overlay(pred_img, label_txt, result.orig_shape, show, num_pred, num_true)
        
    if save_path:
        save_path = save_path + "/" + img_name + ".jpg"
        print(f"==>> Saving {save_path}")
        plt.savefig(save_path)


def _plot_side(img1, img2, show=False, *args):
    num_pred, num_true = args
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    ax1.imshow(img1)
    ax1.axis("off")
    ax1.set_title(f"{num_pred} junctions")
    ax2.imshow(img2)
    ax2.axis("off")
    ax2.set_title(f"{num_true} junctions")
    plt.tight_layout()
    if show:
        plt.show()


def _plot_overlay(pred_img, label_txt, image_shape, show, *args):
    """Plot true junctions as yellow circle on top of predicted image"""
    num_pred, num_true = args
    height, width = image_shape
    
    xywhn_true = []
    with open(label_txt, "r") as f:
        lines = f.readlines()
        for line in lines:
            info = line.split(" ")
            _, x, y, _, _ = int(info[0]), float(info[1]), float(info[2]), float(info[3]), float(info[4])
            xywhn_true.append((x, y))
        
    xywh_true = torch.tensor(xywhn_true) * torch.tensor([width, height])
    
    plt.figure(figsize=(8, 8))
    plt.imshow(pred_img)
    plt.scatter(xywh_true[:, 0], xywh_true[:, 1], marker="o", color="y", s=30)
    plt.title(f"{num_pred} junctions - {num_true} junctions")
    plt.axis("off")
    plt.tight_layout()
            
    if show:
        plt.show()


if __name__ == "__main__":
    val_folder = "data/splits/val/images"
    for filename in os.listdir(val_folder):
        img_path = f"{val_folder}/{filename}"
        conf = 0.289
        mode = "overlay"
        show = True
        save_path = None
        
        visualize_result(
            img_path=img_path,
            conf=conf,
            mode=mode,
            show=show,
            save_path=save_path
        )
        
        break
    