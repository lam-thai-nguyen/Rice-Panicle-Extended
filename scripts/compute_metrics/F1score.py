import os
from tqdm import tqdm
from .compute_f1_score import compute_f1_score, save_as_excel
from .save_f1_score import save_f1_score

class F1score:
    def __init__(self, split_name, mode, conf, iou_threshold=0.1):
        self.split_name = split_name
        self.mode = mode
        self.conf = conf
        self.iou_threshold = iou_threshold
        self.xlsx_path = f'logs/{split_name}/{mode}/f1_score.xlsx'
        self.img_path = f'logs/{split_name}/{mode}/f1_score.png'
        
    def compute_f1_score(self, save_history):
        history = dict()
        images_dir = f"data/splits/{self.split_name}/{self.mode}/images"
        labels_dir = f"data/splits/{self.split_name}/{self.mode}/labels"
        
        print(f"==>> F1score reading {images_dir} |||| {labels_dir}")
        print(f"==>> Saving history? {save_history}")
        
        with tqdm(os.listdir(images_dir), desc="Evaluating") as pbar:
            for filename in pbar:
                # Configuration
                img_path = f"{images_dir}/{filename}"
                label_path = f"{labels_dir}/{filename[:-4]}.txt"
                checkpoint = f"checkpoints/{self.split_name}/best.pt"

                # Compute metrics
                f1, precision, recall = compute_f1_score(img_path, label_path, checkpoint, self.conf, self.iou_threshold)

                # Update progression bar
                pbar.set_postfix({"F1": f"{f1:.2f}", "P": f"{precision:.2f}", "R": f"{recall:.2f}"})
                
                # Update history
                history[filename] = (f1, precision, recall)

        if save_history:
            save_as_excel(history, self.xlsx_path)
            
        print(f"==>> F1score finished computing F1 score")
        
    def save_f1_score(self):
        save_f1_score(self.xlsx_path, self.img_path)
        print(f"==>> F1score finished saving F1 score")

    def compute_save_f1_score(self):
        self.compute_f1_score(save_history=True)
        self.save_f1_score()
        