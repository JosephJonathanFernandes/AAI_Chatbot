from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_confusion_heatmap(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("admission", "eligibility", 3),
        ("admission", "exams", 3),
        ("gratitude", "greetings", 2),
        ("admission", "out_of_scope", 2),
        ("fees", "campus_life", 2),
    ]

    true_labels = ["admission", "gratitude", "fees"]
    pred_labels = ["eligibility", "exams", "greetings", "out_of_scope", "campus_life"]

    matrix = pd.DataFrame(0, index=true_labels, columns=pred_labels, dtype=int)
    for true_label, pred_label, count in pairs:
        matrix.loc[true_label, pred_label] += count

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sns.heatmap(
        matrix,
        cmap="Blues",
        annot=True,
        fmt="d",
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        square=False,
        annot_kws={"fontsize": 9, "fontweight": "bold"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Intent", labelpad=6)
    ax.set_ylabel("True Intent", labelpad=6)
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout(pad=0.6)

    pdf_path = output_dir / "top_misclassified_intents_heatmap.pdf"
    svg_path = output_dir / "top_misclassified_intents_heatmap.svg"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, svg_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    figures_dir = base_dir / "figures"
    pdf_file, svg_file = make_confusion_heatmap(figures_dir)
    print(f"Saved {pdf_file}")
    print(f"Saved {svg_file}")