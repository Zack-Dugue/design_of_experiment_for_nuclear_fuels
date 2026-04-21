import os

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)



def save_dataframe(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)



def query_records_to_dataframe(records):
    return pd.DataFrame(records)



def plot_accuracy_curves(history_df: pd.DataFrame, out_path: str):
    """
    Plot test accuracy vs number of acquired true labels.

    This is the most important benchmark plot because it tells you whether the
    lookahead strategy actually improves downstream performance compared to the
    random and uncertainty baselines.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, sdf in history_df.groupby("strategy"):
        sdf = sdf.sort_values("num_true_labels")
        ax.plot(sdf["num_true_labels"], sdf["test_accuracy"], marker="o", label=strategy)
    ax.set_xlabel("Number of true labels acquired")
    ax.set_ylabel("Test accuracy")
    ax.set_title("MNIST test accuracy vs acquisition budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



def plot_uncertainty_curves(history_df: pd.DataFrame, out_path: str):
    """
    Plot mean pool uncertainty vs number of acquired true labels.

    This is the MNIST analogue of your original uncertainty-reduction score.
    It answers:
    "Does the method reduce disagreement over the pool as it acquires labels?"
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, sdf in history_df.groupby("strategy"):
        sdf = sdf.sort_values("num_true_labels")
        ax.plot(sdf["num_true_labels"], sdf["pool_mean_uncertainty"], marker="o", label=strategy)
    ax.set_xlabel("Number of true labels acquired")
    ax.set_ylabel("Mean pool uncertainty")
    ax.set_title("Mean pool uncertainty vs acquisition budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



def plot_runtime_curves(history_df: pd.DataFrame, out_path: str):
    """
    Plot outer-loop runtime per acquisition step.

    This helps explain the compute trade-off:
    lookahead should usually be more expensive than uncertainty or random, but
    ideally it buys better sample efficiency.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, sdf in history_df.groupby("strategy"):
        sdf = sdf.sort_values("step")
        ax.plot(sdf["step"], sdf["outer_step_seconds"], marker="o", label=strategy)
    ax.set_xlabel("Acquisition step")
    ax.set_ylabel("Seconds")
    ax.set_title("Runtime per acquisition step")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



def plot_query_trajectory(query_df: pd.DataFrame, out_path: str):
    """
    Lightweight query trajectory visualization.

    Since raw MNIST images do not live in an obvious 2D semantic parameter grid,
    we plot two simple summary coordinates of the chosen images:
    - mean pixel intensity
    - pixel intensity standard deviation

    This is not the feature space used by the policy; it is just a compact visual
    summary so you can see whether different methods sample different regions of
    image-space statistics.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for strategy, sdf in query_df.groupby("strategy"):
        sdf = sdf.sort_values("step")
        ax.scatter(sdf["pixel_mean"], sdf["pixel_std"], label=strategy)
        for _, row in sdf.iterrows():
            ax.annotate(str(int(row["step"])), (row["pixel_mean"], row["pixel_std"]), fontsize=8)
    ax.set_xlabel("Pixel mean")
    ax.set_ylabel("Pixel std")
    ax.set_title("Chosen-query trajectory (text = acquisition step)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
