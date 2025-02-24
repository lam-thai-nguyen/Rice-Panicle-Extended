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
    MIN_SIZE, MAX_SIZE, STEP = 22, 46, 4  # Change if needed
    NUM_SPLITS = 7  # Change if needed
    
    bbox_sizes = list(range(MIN_SIZE, MAX_SIZE+1, STEP))
    split_names = [f"split{n}" for n in range(1, NUM_SPLITS+1)]
    
    f1_train, pr_train, rc_train, f1_val, pr_val, rc_val = read_xlsx(split_names)
    assert len(bbox_sizes) == len(f1_train), "Unmatched list lengths"

    # Set global font size, marker size, and line width
    plt.rcParams.update({
        'font.size': 18,
        'lines.linewidth': 2,
        'lines.markersize': 10
    })
    
    
    # Plot training metrics
    if train:
        plt.figure(figsize=(12, 8))
        plt.plot(bbox_sizes, pr_train, label="Precision", marker='o', c='#009E73')
        plt.plot(bbox_sizes, f1_train, label="$F_1$ score", marker='o', c='#E69F00')
        plt.plot(bbox_sizes, rc_train, label="Recall", marker='o', c='#0072B2')
    

        # Add annotations for training metrics
        for i, txt in enumerate(f1_train):
            plt.annotate(f'{txt:.4f}', (bbox_sizes[i], f1_train[i]), textcoords="offset points", xytext=(0,10), ha='center')
        for i, txt in enumerate(pr_train):
            plt.annotate(f'{txt:.4f}', (bbox_sizes[i], pr_train[i]), textcoords="offset points", xytext=(0,10), ha='center')
        for i, txt in enumerate(rc_train):
            plt.annotate(f'{txt:.4f}', (bbox_sizes[i], rc_train[i]), textcoords="offset points", xytext=(0,10), ha='center')

        plt.xlim(bbox_sizes[0]-2, bbox_sizes[-1]+2)
        plt.xticks(range(bbox_sizes[0], bbox_sizes[-1]+1, 4))
        plt.yticks([])
        plt.xlabel("Bounding box size (pixels)", fontsize=18)
        plt.grid(True, linestyle="-")
        # ax1.title("Bounding box sizes comparison on training metrics")
        plt.legend()
        plt.show()

    # Plot validation metrics
    if val:
        plt.figure(figsize=(12, 8))
        plt.plot(bbox_sizes, pr_val, label="Precision", marker='o', c='#009E73')
        plt.plot(bbox_sizes, f1_val, label="$F_1$ score", marker='o', c='#E69F00')
        plt.plot(bbox_sizes, rc_val, label="Recall", marker='o', c='#0072B2')

        # Add annotations for validation metrics
        for i, txt in enumerate(f1_val):
            plt.annotate(f'{txt:.4f}', (bbox_sizes[i], f1_val[i]), textcoords="offset points", xytext=(0,10), ha='center')
        for i, txt in enumerate(pr_val):
            plt.annotate(f'{txt:.4f}', (bbox_sizes[i], pr_val[i]), textcoords="offset points", xytext=(0,10), ha='center')
        for i, txt in enumerate(rc_val):
            plt.annotate(f'{txt:.4f}', (bbox_sizes[i], rc_val[i]), textcoords="offset points", xytext=(0,10), ha='center')

        plt.xlim(bbox_sizes[0]-2, bbox_sizes[-1]+2)
        plt.xticks(range(bbox_sizes[0], bbox_sizes[-1]+1, 4))
        plt.yticks([])
        plt.xlabel("Bounding box size (pixels)", fontsize=18)
        plt.grid(True, linestyle="-")
        # plt.title("Bounding box sizes comparison on evaluation metrics")
        plt.legend()
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

    fig, (ax1, plt) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.set_title("Training metrics")
    plt.set_title("Evaluation metrics")

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
    subplot_metrics(plt, f1_val, pr_val, rc_val)

    fig.suptitle(f"Comparison of Metrics for HBB, OBBv1, OBBv2 at BB Size {fixed_bbox_size}")
    fig.legend(metric_labels, title="Metrics")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_optimal_bbox(train=True, val=True)
    