import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import pandas as pd


def plot_optimal_bbox(train=True, val=True):
    """
    Generate a figure showing the correlation between bbox size and the accuracy metrics

    Args:
        train (bool, optional): whether or not to show training metrics. Defaults to True.
        val (bool, optional): whether or not to show validation metrics. Defaults to True.
    """
    # Data
    bbox_sizes = list(range(22, 82+1, 4))
    split_names = [f"split{n}" for n in range(1, 16+1)]
    
    f1_train, pr_train, rc_train, f1_val, pr_val, rc_val = read_xlsx(split_names)
    assert len(bbox_sizes) == len(f1_train), "Unmatched list lengths"

    if train and val:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    elif train and not val:
        fig, (ax1) = plt.subplots(1, 1, figsize=(8, 7))
    elif not train and val:
        fig, (ax2) = plt.subplots(1, 1, figsize=(8, 7))
    else:
        return
    

    # Plot training metrics
    if train:
        ax1.plot(bbox_sizes, pr_train, label="Precision", marker='o', markersize=8, c='#009E73')
        ax1.plot(bbox_sizes, f1_train, label="$F_1$ score", marker='o', markersize=8, c='#E69F00')
        ax1.plot(bbox_sizes, rc_train, label="Recall", marker='o', markersize=8, c='#0072B2')
    

        # Add annotations for training metrics
        for i, txt in enumerate(f1_train):
            ax1.annotate(f'{txt:.4f}', (bbox_sizes[i], f1_train[i]), textcoords="offset points", xytext=(0,10), ha='center')
        for i, txt in enumerate(pr_train):
            ax1.annotate(f'{txt:.4f}', (bbox_sizes[i], pr_train[i]), textcoords="offset points", xytext=(0,10), ha='center')
        for i, txt in enumerate(rc_train):
            ax1.annotate(f'{txt:.4f}', (bbox_sizes[i], rc_train[i]), textcoords="offset points", xytext=(0,10), ha='center')

        ax1.set_xlim(bbox_sizes[0]-2, bbox_sizes[-1]+2)
        ax1.set_xticks(range(bbox_sizes[0], bbox_sizes[-1]+1, 4))
        # ax1.set_ylim(0.8, 1.0)
        ax1.set_yticks([])
        ax1.set_xlabel("Bounding box size")
        ax1.grid(True, linestyle="-")
        ax1.set_title("Bounding box sizes comparison on training metrics")
        ax1.legend()

    # Plot validation metrics
    if val:
        ax2.plot(bbox_sizes, pr_val, label="Precision", marker='o', markersize=8, c='#009E73')
        ax2.plot(bbox_sizes, f1_val, label="$F_1$ score", marker='o', markersize=8, c='#E69F00')
        ax2.plot(bbox_sizes, rc_val, label="Recall", marker='o', markersize=8, c='#0072B2')

        # Add annotations for validation metrics
        for i, txt in enumerate(f1_val):
            ax2.annotate(f'{txt:.4f}', (bbox_sizes[i], f1_val[i]), textcoords="offset points", xytext=(0,10), ha='center')
        for i, txt in enumerate(pr_val):
            ax2.annotate(f'{txt:.4f}', (bbox_sizes[i], pr_val[i]), textcoords="offset points", xytext=(0,10), ha='center')
        for i, txt in enumerate(rc_val):
            ax2.annotate(f'{txt:.4f}', (bbox_sizes[i], rc_val[i]), textcoords="offset points", xytext=(0,10), ha='center')

        ax2.set_xlim(bbox_sizes[0]-2, bbox_sizes[-1]+2)
        ax2.set_xticks(range(bbox_sizes[0], bbox_sizes[-1]+1, 4))
        # ax2.set_ylim(0.8, 1.0)
        ax2.set_yticks([])
        ax2.set_xlabel("Bounding box size")
        ax2.grid(True, linestyle="-")
        ax2.set_title("Bounding box sizes comparison on evaluation metrics")
        ax2.legend()

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
    
    
def plot_fixed_bbox_comparison(fixed_bbox_size: int):
    # Hard-coded HYPERPARAMETERS
    bbox_versions = ['HBB', 'OBBv1', 'OBBv2']
    split_names = ['split1', 'split2', 'split3']
    
    # Retrieve metrics
    f1_train, pr_train, rc_train, f1_val, pr_val, rc_val = read_xlsx(split_names)
    
    # Define colors for each metric
    metric_colors = {'F1': 'tomato', 'Precision': 'mediumseagreen', 'Recall': 'royalblue'}
    metric_labels = ['F1', 'Precision', 'Recall']
    
    # Define x-axis positions for each subject
    x_values = range(len(bbox_versions))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.set_title("Training metrics")
    ax2.set_title("Evaluation metrics")

    def subplot_metrics(ax, f1, pr, rc):
        for i, (f1, pr, rc) in enumerate(zip(f1, pr, rc)):
            # Plot each metric as a separate point on the same vertical line
            metrics = [f1, pr, rc]
            offsets = [-0.1, 0, 0.1]  # Slight horizontal offsets for clarity

            for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
                ax.plot(x_values[i] + offsets[j], metric, marker='o', markersize=8, color=metric_colors[label], label=label if i == 0 else "")
                ax.annotate(f'{metric:.4f}', (x_values[i] + offsets[j], metric), textcoords="offset points", xytext=(0, 10), ha='center')

            # Connect metrics with a vertical line for each subject
            ax.plot([x_values[i] + offset for offset in offsets], metrics, color='gray', linestyle='--', linewidth=1)
        
        # Set x-axis properties
        ax.set_xticks(x_values)
        ax.set_xticklabels(bbox_versions)
        ax.set_xlim(-0.5, len(bbox_versions) - 0.5)
        ax.set_ylim(0.8, 1.0)
        ax.set_ylabel("Metric Value")

        # Add grid
        ax.grid(True, linestyle="--")
        
    subplot_metrics(ax1, f1_train, pr_train, rc_train)
    subplot_metrics(ax2, f1_val, pr_val, rc_val)

    fig.suptitle(f"Comparison of Metrics for HBB, OBBv1, OBBv2 at BB Size {fixed_bbox_size}")
    fig.legend(metric_labels, title="Metrics")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_optimal_bbox(train=False, val=True)
    