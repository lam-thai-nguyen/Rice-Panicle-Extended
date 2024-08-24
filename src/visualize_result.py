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
    
    split_name = "split9"
    run_name = ""
    plot_loss = False
    predict_show = True
    mode = "train"
    conf = confidence[split_name][0] if mode == "train" else confidence[split_name][1]
    img_name = benchmark_image["lots_of_problems"]
    visual_mode = "side"
    show = True
    save_path = None
    
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
    