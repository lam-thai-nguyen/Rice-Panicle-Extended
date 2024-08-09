import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_f1_score(xlsx_file, save_path):
    # Read the xlsx file and get the columns
    df = pd.read_excel(xlsx_file)
    df.columns = df.columns.str.strip()
    
    columns = ['f1', 'precision', 'recall']
    
    metrics = df[columns]
    
    # Plot
    _, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax1, ax2, ax3 = axes
    
    # Plot for 'f1'
    mean_f1 = metrics['f1'].mean()
    ax1.axvline(mean_f1, color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {mean_f1 * 100:.2f}%')
    
    factor = 1.5  # Change this if needed
    threshold = metrics['f1'].mean() - factor * metrics['f1'].std()
    count = metrics[metrics['f1'] <= threshold]['f1'].count()
    
    # Plot values below threshold
    sns.histplot(metrics[metrics['f1'] <= threshold]['f1'], ax=ax1, color='blue', bins=14, label=f"Error analysis ({count})", alpha=0.2)
    # Plot values above threshold
    sns.histplot(metrics[metrics['f1'] > threshold]['f1'], kde=True, ax=ax1, color='blue', bins=16)
    # Plot threshold
    ax1.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold:.4f}')
    
    ax1.set_xlabel("$F_1$ Score")
    ax1.legend()
    
    # Plot for 'precision'
    sns.histplot(metrics['precision'], kde=True, ax=ax2, color='orange', bins=30)
    ax2.set_xlabel("Precision")
    mean_precision = metrics['precision'].mean()
    ax2.axvline(mean_precision, color='orange', linestyle='dashed', linewidth=2, label=f'Mean: {mean_precision * 100:.2f}%')
    ax2.legend()
    
    # Plot for 'recall'
    sns.histplot(metrics['recall'], kde=True, ax=ax3, color='green', bins=30)
    ax3.set_xlabel("Recall")
    mean_recall = metrics['recall'].mean()
    ax3.axvline(mean_recall, color='green', linestyle='dashed', linewidth=2, label=f'Mean: {mean_recall * 100:.2f}%')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"==>> Saving {save_path}")


if __name__ == "__main__":
    split_name = 'split2'  # Change this if needed
    run_name = 'run3'  # Change this if needed
    mode = 'train'  # Change this if needed
    xlsx_file = f'logs/{split_name}/{run_name}/{mode}/f1_score.xlsx'
    save_path = f'logs/{split_name}/{run_name}/{mode}/f1_score.png'
    plot_f1_score(xlsx_file, save_path)
