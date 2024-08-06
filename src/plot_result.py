# ========================================================================================= #
#     This script plots a 2 by 5 figure showing the cost per iter and metric per iter       #
#     Requirements: results.csv file from training ultralytics model                        #
# ========================================================================================= #

import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def plot_result(csv_file, save_path):
    # Read the csv file and get the columns
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    
    columns = [
    'train/box_om', 'train/cls_om', 'train/dfl_om',
    'train/box_oo', 'train/cls_oo', 'train/dfl_oo',
    'metrics/precision(B)', 'metrics/recall(B)',
    'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
    ]
    
    # Plot
    _, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, col in enumerate(columns):
        ax = axes[i // 5, i % 5]
        ax.plot(df['epoch'], df[col], label="results", alpha=1., marker='o', markersize=3)
        smooth_data = uniform_filter1d(df[col], size=10)
        ax.plot(df['epoch'], smooth_data, label='smooth', linestyle='dotted', linewidth=2)
        ax.set_title(col)
        if i == 1:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"==>> Saving {save_path}")
    

if __name__ == '__main__':
    split_name = 'split2'
    run_name = 'run1'
    mode = 'train'
    path = f'logs/{split_name}/{run_name}/{mode}'
    csv_file = f'{path}/results.csv'
    save_path = f'{path}/results.png'
    plot_result(csv_file, save_path)
    