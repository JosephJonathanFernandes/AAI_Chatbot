from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_suite_composition_chart(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        "In-domain task",
        "Explicit out-of-scope",
        "Robustness/noisy in-domain",
        "Boundary inputs",
    ]
    pass_rates = np.array([86.2, 80.0, 50.0, 0.0])
    remaining = 100.0 - pass_rates

    x = np.arange(len(categories))
    width = 0.36

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    pass_bars = ax.bar(
        x - width / 2,
        pass_rates,
        width,
        color="#1f77b4",
        label="Pass rate",
    )
    rem_bars = ax.bar(
        x + width / 2,
        remaining,
        width,
        color="#b0b0b0",
        label="Remaining",
    )

    for bars in (pass_bars, rem_bars):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylim(0, 106)
    ax.set_ylabel("Percentage")
    ax.set_title("120-Scenario Suite Composition and Pass Rates")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=14, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

    # Requested academic cleanup
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()

    svg_path = output_dir / "suite_composition_pass_rates.svg"
    pdf_path = output_dir / "suite_composition_pass_rates.pdf"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return svg_path, pdf_path


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    svg_file, pdf_file = make_suite_composition_chart(base / "figures")
    print(f"Saved {svg_file}")
    print(f"Saved {pdf_file}")
