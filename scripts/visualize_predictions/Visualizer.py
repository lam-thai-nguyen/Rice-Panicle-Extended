import os
from tqdm import tqdm
from .plot_loss import plot_loss
from .predict_show import predict_show

class Visualizer:
    def __init__(self, split_name, run_name):
        self.split_name = split_name
        self.run_name = run_name
        
    def plot_loss(self):
        path = f'logs/{self.split_name}/{self.run_name}/train' if self.run_name else f'logs/{self.split_name}/train'
        csv_file = f'{path}/results.csv'
        save_path = f'{path}/results.png'
        print(f"==>> Reading {csv_file}")
        plot_loss(csv_file, save_path)
        print("Finished plotting losses")
        
    def predict_show(self, mode, conf, img_name=None, visual_mode="side", show=True, save_path=None):
        """
        mode: train or val
        visual_mode: single or side or overlay
        img_name: [Example] 1_2_1_1_1_DSC01251.jpg -> without path
        """
        mode = mode
        conf = conf
        root_dir = f"data/splits/{self.split_name}/{mode}/images"
        img_name = img_name.split("/")[-1]
        visual_mode = visual_mode
        show = show
        save_path = save_path
        print(f"==>> Reading {root_dir}")
        
        with tqdm(os.listdir(root_dir), desc="Visualizing") as pbar:
            if img_name:
                for filename in pbar:
                    if filename == img_name:
                        # Configuration
                        img_path = f"{root_dir}/{filename}"
                        checkpoint = f"checkpoints/{self.split_name}/{self.run_name}/best.pt" if self.run_name else f"checkpoints/{self.split_name}/best.pt"
                        
                        # Get results
                        predict_show(
                            img_path=img_path,
                            checkpoint=checkpoint,
                            conf=conf,
                            mode=visual_mode,
                            show=show,
                            save_path=save_path
                        )
                        
                        # Update progress bar
                        pbar.set_postfix({"mode": mode, "conf": conf})
                        break
                    
            else:
                for filename in pbar:
                    # Configuration
                    img_path = f"{root_dir}/{filename}"
                    checkpoint = f"checkpoints/{self.split_name}/{self.run_name}/best.pt" if self.run_name else f"checkpoints/{self.split_name}/best.pt"
                    
                    # Get results
                    predict_show(
                        img_path=img_path,
                        checkpoint=checkpoint,
                        conf=conf,
                        mode=visual_mode,
                        show=show,
                        save_path=save_path
                    )
                    
                    # Update progress bar
                    pbar.set_postfix({"mode": mode, "conf": conf})
                        
        print(f"Finished visualizing {img_name}") if img_name else print("Finished visualizing all images")
        