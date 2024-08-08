import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_f1_score(xlsx_file, save_path):
    # Read the xlsx file and get the columns
    df = pd.read_excel(xlsx_file)
    df.columns = df.columns.str.strip()
    
    columns = ['f1', 'precision', 'recall', 'abs(pr-rc)']
    
    metrics = df[columns]
    
    # Plot
    _, axes = plt.subplots(1, 4, figsize=(16, 6))
    ax1, ax2, ax3, ax4 = axes
    
    # Plot for 'f1'
    sns.histplot(metrics['f1'], kde=True, ax=ax1, color='blue', bins=30)
    mean_f1 = metrics['f1'].mean()
    ax1.axvline(mean_f1, color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {mean_f1 * 100:.2f}%')
    ax1.legend()
    
    # Plot for 'precision'
    sns.histplot(metrics['precision'], kde=True, ax=ax2, color='orange', bins=30)
    mean_precision = metrics['precision'].mean()
    ax2.axvline(mean_precision, color='orange', linestyle='dashed', linewidth=2, label=f'Mean: {mean_precision * 100:.2f}%')
    ax2.legend()
    
    # Plot for 'recall'
    sns.histplot(metrics['recall'], kde=True, ax=ax3, color='green', bins=30)
    mean_recall = metrics['recall'].mean()
    ax3.axvline(mean_recall, color='green', linestyle='dashed', linewidth=2, label=f'Mean: {mean_recall * 100:.2f}%')
    ax3.legend()
    
    # Plot for 'abs(pr-rc)' with threshold
    factor = 2.0  # Change this if needed
    threshold = metrics['abs(pr-rc)'].mean() + factor * metrics['abs(pr-rc)'].std()

    # Plot values below threshold
    sns.histplot(metrics[metrics['abs(pr-rc)'] <= threshold]['abs(pr-rc)'], kde=True, ax=ax4, color='red', bins=20, alpha=0.1)
    
    # Plot values above threshold
    sns.histplot(metrics[metrics['abs(pr-rc)'] > threshold]['abs(pr-rc)'], kde=True, ax=ax4, color='red', bins=10, label="Error analysis")

    ax4.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold:.4f}')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"==>> Saving {save_path}")


if __name__ == "__main__":
    split_name = 'split2'  # Change this if needed
    run_name = 'run3'  # Change this if needed
    mode = 'val'  # Change this if needed
    xlsx_file = f'logs/{split_name}/{run_name}/{mode}/f1_score.xlsx'
    save_path = f'logs/{split_name}/{run_name}/{mode}/f1_score.png'
    plot_f1_score(xlsx_file, save_path)
