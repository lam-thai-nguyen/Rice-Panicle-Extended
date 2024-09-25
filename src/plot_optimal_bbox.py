import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # Data
    x_value = [14, 18, 22, 26, 30, 34, 38, 42, 46, 50]  # bbox sizes
    split_names = ['split20', 'split21', 'split22', 'split15', 'split16', 'split17', 'split18', 'split19', 'split23', 'split24']
    
    f1_train, pr_train, rc_train, f1_val, pr_val, rc_val = read_xlsx(split_names)
    assert len(x_value) == len(f1_train), "Unmatched list lengths"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot training metrics
    ax1.plot(x_value, f1_train, label="f1_train", marker='o', markersize=8, c='tomato')
    ax1.plot(x_value, pr_train, label="pr_train", marker='o', markersize=8, c='mediumseagreen')
    ax1.plot(x_value, rc_train, label="rc_train", marker='o', markersize=8, c='royalblue')

    # Add annotations for training metrics
    for i, txt in enumerate(f1_train):
        ax1.annotate(f'{txt:.4f}', (x_value[i], f1_train[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i, txt in enumerate(pr_train):
        ax1.annotate(f'{txt:.4f}', (x_value[i], pr_train[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i, txt in enumerate(rc_train):
        ax1.annotate(f'{txt:.4f}', (x_value[i], rc_train[i]), textcoords="offset points", xytext=(0,10), ha='center')

    ax1.set_xlim(x_value[0]-2, x_value[-1]+2)
    ax1.set_xticks(range(x_value[0], x_value[-1]+1, 4))
    ax1.set_ylim(0.8, 1.0)
    ax1.set_yticks([])
    ax1.set_xlabel("Bounding box size")
    ax1.grid(True, linestyle="--")
    ax1.set_title("Training Metrics")
    ax1.legend()

    # Plot validation metrics
    ax2.plot(x_value, f1_val, label="f1_val", marker='o', markersize=8, c='tomato')
    ax2.plot(x_value, pr_val, label="pr_val", marker='o', markersize=8, c='mediumseagreen')
    ax2.plot(x_value, rc_val, label="rc_val", marker='o', markersize=8, c='royalblue')

    # Add annotations for validation metrics
    for i, txt in enumerate(f1_val):
        ax2.annotate(f'{txt:.4f}', (x_value[i], f1_val[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i, txt in enumerate(pr_val):
        ax2.annotate(f'{txt:.4f}', (x_value[i], pr_val[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i, txt in enumerate(rc_val):
        ax2.annotate(f'{txt:.4f}', (x_value[i], rc_val[i]), textcoords="offset points", xytext=(0,10), ha='center')

    ax2.set_xlim(x_value[0]-2, x_value[-1]+2)
    ax2.set_xticks(range(x_value[0], x_value[-1]+1, 4))
    ax2.set_ylim(0.8, 1.0)
    ax2.set_yticks([])
    ax2.set_xlabel("Bounding box size")
    ax2.grid(True, linestyle="--")
    ax2.set_title("Evaluation Metrics")
    ax2.legend()

    fig.suptitle("Finding Optimal Bounding Box Size")

    plt.tight_layout()
    plt.show()


def read_xlsx(split_names: list):
    f1_train, pr_train, rc_train, f1_val, pr_val, rc_val = [], [], [], [], [], []
    
    for split_name in split_names:
        train_path = f"logs/{split_name}/train/f1_score.xlsx"
        val_path = f"logs/{split_name}/val/f1_score.xlsx"
        
        train_df = pd.read_excel(train_path)
        val_df = pd.read_excel(val_path)
        
        f1_train.append(train_df['f1'].mean())
        pr_train.append(train_df['precision'].mean())
        rc_train.append(train_df['recall'].mean())
        
        f1_val.append(val_df['f1'].mean())
        pr_val.append(val_df['precision'].mean())
        rc_val.append(val_df['recall'].mean())
        
    return f1_train, pr_train, rc_train, f1_val, pr_val, rc_val
    
    
if __name__ == "__main__":
    main()
    