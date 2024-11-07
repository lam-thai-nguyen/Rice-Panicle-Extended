import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt


def predict_show(img_path, checkpoint, conf):
    """
    This function shows a yolo prediction image with some extra info.
    
    Args:
        img_path (str): image path
        checkpoint (str): model path
        conf (float): confidence cutoff
    """
    # Inference
    model = YOLO(checkpoint)

    results = model.predict(source=img_path, conf=conf)
    result = results[0]
    
    if result.boxes is not None:
        num_pred = len(result.boxes)  
        flag = "HBB"
    else: 
        num_pred = len(result.obb)
        flag = "OBB"

    # Plotting config.
    pred_img = result.plot(
        line_width=2,
        # conf=True,
        # labels=True,
        # font_size=10.,
    )
    
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

    # Plot
    plt.figure(figsize=(8, 9))
    plt.imshow(pred_img)
    plt.axis("off")
    plt.tight_layout()
    plt.title(f"{flag} | {num_pred} junctions")
    plt.show()
    