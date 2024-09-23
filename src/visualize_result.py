import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.visualize_predictions.Visualizer import Visualizer
import yaml


def main(split_name, run_name, plot_loss=False, predict_show=False, mode="train", conf=None, img_name=None, visual_mode="side", show=True, save_path=None):
    visualizer = Visualizer(
        split_name=split_name,
        run_name=run_name,
    )
    
    if plot_loss:
        visualizer.plot_loss()
    
    if predict_show:
        visualizer.predict_show(
            mode=mode,
            conf=conf,
            img_name=img_name,
            visual_mode=visual_mode,
            show=show,
            save_path=save_path
        )
    

if __name__ == "__main__":
    with open("src/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    confidence = config["confidence"]
    benchmark_image = config["benchmark_image"]
    
    split_name = "split16"  # Change this if needed
    run_name = ""  # Change this if needed
    plot_loss = False  # Change this if needed
    predict_show = True  # Change this if needed
    mode = "train"  # Change this if needed
    conf = confidence[split_name][0] if mode == "train" else confidence[split_name][1]
    img_name = benchmark_image["lots_of_junctions"]  # Change this if needed
    visual_mode = "side"  # Change this if needed
    show = True  # Change this if needed
    save_path = None  # Change this if needed
    
    main(
        split_name=split_name,
        run_name=run_name,
        plot_loss=plot_loss,
        predict_show=predict_show,
        mode=mode,
        conf=conf,
        img_name=img_name,
        visual_mode=visual_mode,
        show=show,
        save_path=save_path
    )
    