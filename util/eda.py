import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def show_data_correlations(input_df, dataset_label):
    matrix = input_df.corr()
    plt.figure(figsize=(15, 10))
    mask = np.triu(matrix)
    sns.heatmap(matrix, annot=True, fmt=".2f",mask=mask)
    plt.title(f"{dataset_label} Dataset Correlation Heatmap")
    plt.show()


def plot_null_values(data_frame):
    plt.figure(figsize=(15, 10))
    sns.heatmap(data_frame.isna(), cmap='plasma', cbar=False)
    plt.title('Heatmap of Null Values')
    plt.show()


def summarize_statistics(data_frame):
    with pd.option_context('display.max_columns', None):
        stats_summary = data_frame.describe().T
        stats_summary['kurt'] = data_frame.kurt()
        stats_summary['skew'] = data_frame.skew()
        print("\nSummary of Statistics:")
        print(stats_summary)


def compute_group_means(data_frame, group_cols, target_col):
    for col in group_cols:
        means = data_frame.groupby(col)[target_col].mean()
        print(f"\nMean {target_col} by {col}:")
        print(means)


def visualize_outliers(data_frame):
    num_cols = data_frame.select_dtypes(include=['int64', 'float64']).columns
    count_cols = len(num_cols)
    fig, axes = plt.subplots(ncols=count_cols, nrows=1, figsize=(5 * count_cols, 8))

    if count_cols == 1:
        axes = [axes]

    for idx, col_name in enumerate(num_cols):
        sns.boxplot(data_frame[col_name], ax=axes[idx])
        axes[idx].set_title(f'Boxplot of {col_name}')

    plt.tight_layout()
    plt.show()