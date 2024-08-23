import os
from tqdm import tqdm
from .compute_f1_score import compute_f1_score, save_as_excel
from .save_f1_score import save_f1_score

class F1score:
    def __init__(self, split_name, run_name, mode, conf, iou_threshold):
        self.split_name = split_name
        self.run_name = run_name
        self.mode = mode
        self.conf = conf
        self.iou_threshold = iou_threshold
        self.xlsx_path = f'logs/{split_name}/{run_name}/{mode}/f1_score.xlsx' if run_name else f'logs/{split_name}/{mode}/f1_score.xlsx'
        self.img_path = f'logs/{split_name}/{run_name}/{mode}/f1_score.png' if run_name else f'logs/{split_name}/{mode}/f1_score.png'
        
    def compute_f1_score(self, save_history):
        history = dict()
        root_dir = f"data/splits/{self.split_name}/{self.mode}/images"
        
        print(f"==>> Reading {root_dir}")
        print(f"==>> Saving history? {save_history}")
        
        with tqdm(os.listdir(root_dir), desc="Evaluating") as pbar:
            for filename in pbar:
                # Configuration
                img_path = f"{root_dir}/{filename}"
                checkpoint = f"checkpoints/{self.split_name}/{self.run_name}/best.pt" if self.run_name else f"checkpoints/{self.split_name}/best.pt"

                # Compute metrics
                f1, precision, recall = compute_f1_score(img_path, checkpoint, self.conf, self.iou_threshold)

                # Update progression bar
                pbar.set_postfix({"F1": f"{f1:.2f}", "P": f"{precision:.2f}", "R": f"{recall:.2f}"})
                
                # Update history
                history[filename] = (f1, precision, recall)

        if save_history:
            save_as_excel(history, self.xlsx_path)
            
        print(f"Finished computing F1 score")
        
    def save_f1_score(self):
        print(f"==>> Reading {self.xlsx_path}")
        print(f"==>> Saving {self.img_path}")
        save_f1_score(self.xlsx_path, self.img_path)
        print(f"Finished saving F1 score")

    def compute_save_f1_score(self):
        self.compute_f1_score(save_history=True)
        self.save_f1_score()