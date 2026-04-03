# ============================================================
# FILE: preprocessing.py
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# ── Global plot style ────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "figure.facecolor": "white"})


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_class_distribution(y_raw, y_remapped):
    """Bar charts showing label distribution before and after remapping."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Class Distribution", fontsize=14, fontweight="bold")

    # Before remapping
    raw_counts = y_raw.value_counts().sort_index()
    axes[0].bar(raw_counts.index.astype(str), raw_counts.values,
                color=["#4C72B0", "#DD8452"], edgecolor="white", width=0.5)
    axes[0].set_title("Raw Labels (from UCI)")
    axes[0].set_xlabel("Label Value")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(raw_counts.values):
        axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

    # After remapping
    remap_counts = y_remapped.value_counts().sort_index()
    labels = [f"{idx} ({'Legitimate' if idx == 0 else 'Phishing'})"
              for idx in remap_counts.index]
    axes[1].bar(labels, remap_counts.values,
                color=["#4C72B0", "#DD8452"], edgecolor="white", width=0.5)
    axes[1].set_title("Remapped Labels (0=Legit, 1=Phishing)")
    axes[1].set_xlabel("Label")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(remap_counts.values):
        axes[1].text(i, v + 50, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("01_class_distribution.png", bbox_inches="tight")
    plt.show()
    print("  Saved → 01_class_distribution.png")


def plot_feature_distributions(X: pd.DataFrame):
    """
    Grid of histograms for all 31 features.
    Shows the -1 / 0 / 1 ternary nature of UCI features.
    """
    n_features = X.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 2.8))
    fig.suptitle("Feature Value Distributions (All 31 Features)",
                 fontsize=14, fontweight="bold", y=1.01)
    axes_flat = axes.flatten()

    for i, col in enumerate(X.columns):
        counts = X[col].value_counts().sort_index()
        axes_flat[i].bar(counts.index.astype(str), counts.values,
                         color="#4C72B0", edgecolor="white")
        axes_flat[i].set_title(col, fontsize=8, fontweight="bold")
        axes_flat[i].set_xlabel("Value", fontsize=7)
        axes_flat[i].set_ylabel("Count", fontsize=7)
        axes_flat[i].tick_params(labelsize=7)

    # Hide any unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("02_feature_distributions.png", bbox_inches="tight")
    plt.show()
    print("  Saved → 02_feature_distributions.png")


def plot_correlation_heatmap(X: pd.DataFrame):
    """Pearson correlation heatmap of all 31 features."""
    fig, ax = plt.subplots(figsize=(14, 11))
    corr = X.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle mask
    sns.heatmap(
        corr,
        mask=mask,
        annot=False,
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax
    )
    ax.set_title("Feature Correlation Matrix (Lower Triangle)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)

    plt.tight_layout()
    plt.savefig("03_correlation_heatmap.png", bbox_inches="tight")
    plt.show()
    print("  Saved → 03_correlation_heatmap.png")


def plot_feature_vs_label(X: pd.DataFrame, y: pd.Series):
    """
    Stacked bar chart per feature showing value distribution
    split by class (Legitimate vs Phishing).
    Helps identify discriminative features visually.
    """
    df = X.copy()
    df["label"] = y.values

    n_features = X.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3))
    fig.suptitle("Feature Values by Class (Legitimate vs Phishing)",
                 fontsize=14, fontweight="bold", y=1.01)
    axes_flat = axes.flatten()

    for i, col in enumerate(X.columns):
        ct = df.groupby([col, "label"]).size().unstack(fill_value=0)
        ct.columns = ["Legitimate" if c == 0 else "Phishing"
                      for c in ct.columns]
        ct.plot(kind="bar", ax=axes_flat[i], color=["#4C72B0", "#DD8452"],
                edgecolor="white", width=0.6)
        axes_flat[i].set_title(col, fontsize=8, fontweight="bold")
        axes_flat[i].set_xlabel("Feature Value", fontsize=7)
        axes_flat[i].set_ylabel("Count", fontsize=7)
        axes_flat[i].tick_params(labelsize=7)
        axes_flat[i].get_legend().remove()
        axes_flat[i].tick_params(axis="x", rotation=0)

    # Shared legend
    handles = [
        plt.Rectangle((0,0),1,1, color="#4C72B0", label="Legitimate"),
        plt.Rectangle((0,0),1,1, color="#DD8452", label="Phishing")
    ]
    fig.legend(handles=handles, loc="upper right",
               fontsize=9, title="Class", framealpha=0.9)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("04_feature_vs_label.png", bbox_inches="tight")
    plt.show()
    print("  Saved → 04_feature_vs_label.png")


def plot_scaling_effect(X_raw: pd.DataFrame,
                        X_train_scaled: np.ndarray,
                        feature_names: list):
    """
    Side-by-side comparison of 6 sample features before and after MinMax scaling.
    Demonstrates the preprocessing step visually.
    """
    sample_features = feature_names[:6]
    sample_idx      = [feature_names.index(f) for f in sample_features]

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle("MinMax Scaling Effect — Sample of 6 Features",
                 fontsize=13, fontweight="bold")

    for col_i, (feat, idx) in enumerate(zip(sample_features, sample_idx)):
        # Before scaling
        axes[0, col_i].hist(X_raw.iloc[:, idx], bins=5,
                            color="#4C72B0", edgecolor="white")
        axes[0, col_i].set_title(feat, fontsize=7, fontweight="bold")
        if col_i == 0:
            axes[0, col_i].set_ylabel("Before Scaling", fontsize=9,
                                       fontweight="bold", color="#4C72B0")

        # After scaling
        axes[1, col_i].hist(X_train_scaled[:, idx], bins=10,
                            color="#55A868", edgecolor="white")
        if col_i == 0:
            axes[1, col_i].set_ylabel("After MinMax Scaling", fontsize=9,
                                       fontweight="bold", color="#55A868")

    for ax in axes.flatten():
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig("05_scaling_effect.png", bbox_inches="tight")
    plt.show()
    print("  Saved → 05_scaling_effect.png")


def plot_train_test_split(y_train: pd.Series, y_test: pd.Series):
    """Grouped bar chart confirming stratified split balance."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Train / Test Split — Class Balance Check",
                 fontsize=13, fontweight="bold")

    for ax, (split_name, split_y) in zip(
            axes, [("Training Set (80%)", y_train),
                   ("Test Set (20%)",     y_test)]):
        counts = split_y.value_counts().sort_index()
        bars = ax.bar(
            ["Legitimate\n(0)", "Phishing\n(1)"],
            counts.values,
            color=["#4C72B0", "#DD8452"],
            edgecolor="white", width=0.5
        )
        ax.set_title(split_name, fontweight="bold")
        ax.set_ylabel("Sample Count")
        total = counts.sum()
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 30, f"{v}\n({v/total*100:.1f}%)",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(0, counts.max() * 1.2)

    plt.tight_layout()
    plt.savefig("06_train_test_split.png", bbox_inches="tight")
    plt.show()
    print("  Saved → 06_train_test_split.png")


# ============================================================
# MAIN PIPELINE
# ============================================================

def load_and_preprocess():
    print("=" * 55)
    print("  PHISHING DATASET — PREPROCESSING PIPELINE")
    print("=" * 55)

    # ── 1. Fetch ─────────────────────────────────────────────
    print("\n[1/5] Fetching dataset from UCI ML Repository...")
    phishing_websites = fetch_ucirepo(id=327)
    X = phishing_websites.data.features
    y_raw = phishing_websites.data.targets.squeeze()

    print(f"      Raw label values    : {sorted(y_raw.unique())}")
    print(f"      Raw class counts    :\n{y_raw.value_counts()}")

    # ── 2. Remap labels ──────────────────────────────────────
    print("\n[2/5] Remapping labels...")
    if set(y_raw.unique()) == {1, -1}:
        y = y_raw.map({1: 0, -1: 1})
    elif set(y_raw.unique()) == {1, 0}:
        y = y_raw.copy()
    else:
        raise ValueError(f"Unexpected label values: {y_raw.unique()}")

    # ── 3. Sanity checks ─────────────────────────────────────
    print(f"\n[3/5] Dataset summary:")
    print(f"      Shape            : {X.shape}")
    print(f"      Missing values   : {X.isnull().sum().sum()}")
    print(f"      Remapped counts  :\n{y.value_counts()}")

    # ── 4. Split → Scale ─────────────────────────────────────
    print("\n[4/5] Splitting (80/20 stratified) and scaling...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"      Train : {X_train_scaled.shape[0]} samples")
    print(f"      Test  : {X_test_scaled.shape[0]} samples")

    # ── 5. Visualizations ────────────────────────────────────
    print("\n[5/5] Generating visualizations...")
    feature_names = list(X.columns)

    plot_class_distribution(y_raw, y)
    plot_feature_distributions(X)
    plot_correlation_heatmap(X)
    plot_feature_vs_label(X, y)
    plot_scaling_effect(X, X_train_scaled, feature_names)
    plot_train_test_split(y_train, y_test)

    print("\n" + "=" * 55)
    print("  PREPROCESSING COMPLETE")
    print("=" * 55)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess()