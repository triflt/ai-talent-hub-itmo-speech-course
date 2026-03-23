from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ASSIGNMENT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ASSIGNMENT_DIR / "outputs"
ARTIFACTS_DIR = OUTPUTS_DIR / "report_artifacts"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_beam_plot() -> Path:
    df = load_csv(OUTPUTS_DIR / "task2_beam_sweep" / "summary.csv")
    df["beam_width"] = df["beam_width"].astype(int)
    df["wer"] = df["wer"].astype(float)
    df["cer"] = df["cer"].astype(float)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df["beam_width"], df["wer"], marker="o", label="WER")
    ax1.plot(df["beam_width"], df["cer"], marker="s", label="CER")
    ax1.set_xlabel("Beam width")
    ax1.set_ylabel("Error rate, %")
    ax1.set_title("Beam width vs quality on LibriSpeech")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig.tight_layout()

    out = ARTIFACTS_DIR / "beam_width_vs_quality.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def save_temperature_libri_plot() -> Path:
    df = load_csv(OUTPUTS_DIR / "task3_temperature_sweep" / "summary.csv")
    df["temperature"] = df["temperature"].astype(float)
    df["wer"] = df["wer"].astype(float)
    df["cer"] = df["cer"].astype(float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["temperature"], df["wer"], marker="o", label="WER")
    ax.plot(df["temperature"], df["cer"], marker="s", label="CER")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Error rate, %")
    ax.set_title("Temperature sweep for greedy on LibriSpeech")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = ARTIFACTS_DIR / "temperature_librispeech_greedy.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def save_heatmap(summary_path: Path, title: str, out_name: str) -> Path:
    df = load_csv(summary_path)
    df["alpha"] = df["alpha"].astype(float)
    df["beta"] = df["beta"].astype(float)
    df["wer"] = df["wer"].astype(float)
    pivot = df.pivot(index="alpha", columns="beta", values="wer").sort_index()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xlabel("beta")
    ax.set_ylabel("alpha")
    ax.set_title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax, label="WER, %")
    fig.tight_layout()

    out = ARTIFACTS_DIR / out_name
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def save_domain_bar_plot() -> Path:
    df = load_csv(OUTPUTS_DIR / "task7_domain_eval" / "summary.csv")
    df["wer"] = df["wer"].astype(float)

    pivot = df.pivot(index="method", columns="dataset", values="wer").loc[
        ["greedy", "beam", "beam_lm", "beam_lm_rescore"]
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(pivot.index))
    width = 0.35
    ax.bar([i - width / 2 for i in x], pivot["librispeech_test_other"], width=width, label="LibriSpeech")
    ax.bar([i + width / 2 for i in x], pivot["earnings22_test"], width=width, label="Earnings22")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Greedy", "Beam", "Beam+LM", "Beam+Rescore"])
    ax.set_ylabel("WER, %")
    ax.set_title("Domain shift: WER across datasets")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = ARTIFACTS_DIR / "domain_shift_wer.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def save_temperature_earnings_plot() -> Path:
    df = load_csv(OUTPUTS_DIR / "task7b_temperature_eval" / "summary.csv")
    df["temperature"] = df["temperature"].astype(float)
    df["wer"] = df["wer"].astype(float)

    fig, ax = plt.subplots(figsize=(7, 4))
    for method, label in [("greedy", "Greedy"), ("beam_lm", "Beam + LM")]:
        part = df[df["method"] == method].sort_values("temperature")
        ax.plot(part["temperature"], part["wer"], marker="o", label=label)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("WER, %")
    ax.set_title("Temperature sweep on Earnings22")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = ARTIFACTS_DIR / "temperature_earnings22.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def save_lm_comparison_plot() -> Path:
    df = load_csv(OUTPUTS_DIR / "task9_compare_lms" / "summary.csv")
    df["wer"] = df["wer"].astype(float)

    labels = []
    values = []
    for dataset in ["librispeech_test_other", "earnings22_test"]:
        for lm_name in ["librispeech_3gram", "financial_3gram"]:
            for method in ["beam_lm", "beam_lm_rescore"]:
                row = df[(df["dataset"] == dataset) & (df["lm_name"] == lm_name) & (df["method"] == method)].iloc[0]
                labels.append(f"{dataset}\n{lm_name}\n{method}")
                values.append(row["wer"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("WER, %")
    ax.set_title("LM comparison across domains")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = ARTIFACTS_DIR / "lm_comparison_wer.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def generate_qualitative_examples() -> Path:
    beam = load_csv(OUTPUTS_DIR / "task7_domain_eval" / "librispeech_test_other_beam.csv")
    sf = load_csv(OUTPUTS_DIR / "task7_domain_eval" / "librispeech_test_other_beam_lm.csv")
    rs = load_csv(OUTPUTS_DIR / "task7_domain_eval" / "librispeech_test_other_beam_lm_rescore.csv")

    merged = beam.merge(sf, on="path", suffixes=("_beam", "_sf")).merge(
        rs, on="path", suffixes=("", "_rs")
    )
    merged = merged.rename(
        columns={
            "reference_beam": "reference",
            "hypothesis_beam": "beam_hyp",
            "wer_beam": "beam_wer",
            "cer_beam": "beam_cer",
            "hypothesis_sf": "sf_hyp",
            "wer_sf": "sf_wer",
            "cer_sf": "sf_cer",
            "hypothesis": "rs_hyp",
            "wer": "rs_wer",
            "cer": "rs_cer",
        }
    )
    merged = merged.drop(columns=["reference_sf", "reference_rs"], errors="ignore")

    for col in ["beam_wer", "sf_wer", "rs_wer", "beam_cer", "sf_cer", "rs_cer"]:
        merged[col] = merged[col].astype(float)

    changed = merged[
        (merged["beam_hyp"] != merged["sf_hyp"]) |
        (merged["beam_hyp"] != merged["rs_hyp"])
    ].copy()
    changed["best_lm_wer"] = changed[["sf_wer", "rs_wer"]].min(axis=1)
    changed["improvement"] = changed["beam_wer"] - changed["best_lm_wer"]
    improved = changed[changed["improvement"] > 0].sort_values(
        ["improvement", "beam_wer"], ascending=[False, False]
    )
    disagreements = changed[
        (changed["sf_hyp"] != changed["rs_hyp"]) &
        (~changed["path"].isin(improved["path"]))
    ].sort_values(["beam_wer", "sf_wer", "rs_wer"], ascending=[False, False, False])

    selected = pd.concat([improved.head(6), disagreements.head(2)], ignore_index=True)
    out = ARTIFACTS_DIR / "qualitative_examples.md"
    with out.open("w", encoding="utf-8") as handle:
        handle.write("## Qualitative examples\n\n")
        for idx, row in enumerate(selected.itertuples(index=False), start=1):
            handle.write(f"### Example {idx}\n\n")
            handle.write(f"REF:  {row.reference}\n")
            handle.write(f"BEAM: {row.beam_hyp}\n")
            handle.write(f"SF:   {row.sf_hyp}\n")
            handle.write(f"RS:   {row.rs_hyp}\n")
            handle.write(
                f"WER:  beam={row.beam_wer:.3f}, sf={row.sf_wer:.3f}, rs={row.rs_wer:.3f}\n\n"
            )
    return out


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    generated = [
        save_beam_plot(),
        save_temperature_libri_plot(),
        save_heatmap(OUTPUTS_DIR / "task4_shallow_fusion" / "summary.csv", "Shallow fusion WER heatmap", "task4_shallow_fusion_heatmap.png"),
        save_heatmap(OUTPUTS_DIR / "task6_rescoring" / "summary.csv", "Rescoring WER heatmap", "task6_rescoring_heatmap.png"),
        save_domain_bar_plot(),
        save_temperature_earnings_plot(),
        save_lm_comparison_plot(),
        generate_qualitative_examples(),
    ]
    for path in generated:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
