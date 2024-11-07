import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.visualize_predictions.Visualizer import Visualizer
import yaml


def visualize(plot_loss: False, predict_show=False, **kwargs) -> None:
    split_name = kwargs.get("split_name", None)
    img_path = kwargs.get("img_path", None)
    checkpoint = kwargs.get("checkpoint", None)
    conf = kwargs.get("conf", None)
    
    visualizer = Visualizer()

    if plot_loss:
        visualizer.plot_loss(split_name)

    if predict_show:
        visualizer.predict_show(img_path, checkpoint, conf)


if __name__ == "__main__":
    with open("src/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    confidence = config["confidence"]
    benchmark_image = config["benchmark_image"]
    
    # Hyperparameters (change if needed)
    plot_loss = False
    predict_show = True
    
    # Keyword Arguments (Change if static)
    split_name = "split2"  # static
    
    img_name = benchmark_image["lots_of_junctions"]  # or species/name.jpg
    img_path = f"data/raw/{img_name}"
    
    checkpoint = f"checkpoints/{split_name}/best.pt"

    mode = "train"  # static
    conf = confidence[split_name][0] if mode == "train" else confidence[split_name][1]
    
    visualize(plot_loss, predict_show, split_name=split_name, img_path=img_path, checkpoint=checkpoint, conf=conf)

    