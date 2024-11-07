from .plot_loss import plot_loss
from .predict_show import predict_show

class Visualizer:
    def __init__(self) -> None:
        pass
    
    def plot_loss(self, split_name):
        path = f'logs/{split_name}/train'
        csv_file = f'{path}/results.csv'
        save_path = f'{path}/results.png'
        plot_loss(csv_file, save_path)
        print("==>> Visualizer finished plotting losses.")
        
    def predict_show(self, img_path, checkpoint, conf):
        predict_show(img_path, checkpoint, conf)                
        print("==>> Visualizer finished visualizing.")
        