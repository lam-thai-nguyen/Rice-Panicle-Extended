import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.visualize_predictions.Visualizer import Visualizer


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
    # ============================================= #
    #                   HINTS                       #
    # ============================================= #
    conf_helpers = {  
        "split4": [0.304, 0.309],
        "split5": [0.275, 0.270],  # train, val confs
        "split6": [0.323, 0.322],
        "split7": [0.294, 0.292],
        "split8": [0.233, 0.233],
        "split9": [0.310, 0.310],  
    }
    freq_used = {
        "lots_of_problems": "43_2_1_1_2_DSC09658_.jpg",
        "lots_of_junctions": "37_2_1_1_1_DSC01221.jpg",
    }
    
    split_name = "split8"
    run_name = ""
    plot_loss = False
    predict_show = True
    mode = "train"
    conf = 0.233
    img_name = "43_2_1_1_2_DSC09658_.jpg"
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
    