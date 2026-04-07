from pathlib import Path

import matplotlib.pyplot as plt


def make_latency_breakdown_chart(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    local_nlu_ms = 60.62
    external_llm_ms = 4668.58
    total_ms = 4729.2

    colors = {
        "local": "#1f77b4",
        "external": "#7f7f7f",
        "total": "#111111",
    }

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=(6.8, 2.35))

    ax.barh([0], [local_nlu_ms], color=colors["local"], height=0.46, label="Local NLU Processing")
    ax.barh([0], [external_llm_ms], left=[local_nlu_ms], color=colors["external"], height=0.46,
            label="External LLM Inference & Overhead")

    ax.annotate(f"{local_nlu_ms:.2f} ms", xy=(local_nlu_ms / 2, 0), xytext=(0, 0),
                textcoords="offset points", ha="center", va="center", color="white",
                fontsize=8, fontweight="bold")
    ax.annotate(f"{external_llm_ms:.2f} ms", xy=(local_nlu_ms + external_llm_ms / 2, 0), xytext=(0, 0),
                textcoords="offset points", ha="center", va="center", color="white",
                fontsize=8, fontweight="bold")
    ax.annotate(f"Total: {total_ms:.1f} ms", xy=(total_ms, 0), xytext=(6, 10),
                textcoords="offset points", ha="left", va="bottom", color=colors["total"],
                fontsize=8.5, fontweight="bold")

    ax.set_yticks([0])
    ax.set_yticklabels(["Per-turn latency"])
    ax.set_xlabel("Latency (ms)")
    ax.set_xlim(0, total_ms * 1.08)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2, frameon=False)
    fig.tight_layout(pad=0.5)

    pdf_path = output_dir / "per_turn_latency_breakdown.pdf"
    svg_path = output_dir / "per_turn_latency_breakdown.svg"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, svg_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    figures_dir = base_dir / "figures"
    pdf_path, svg_path = make_latency_breakdown_chart(figures_dir)
    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")