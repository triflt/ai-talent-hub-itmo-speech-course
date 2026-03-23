import argparse
import csv
from pathlib import Path

from evaluate_decoder import ASSIGNMENT_DIR, evaluate_manifest, save_predictions
from wav2vec2decoder import Wav2Vec2Decoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-base-100h")
    parser.add_argument("--lm-model-path", type=str, default="lm/3-gram.pruned.1e-7.arpa.gz")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=ASSIGNMENT_DIR / "outputs" / "task7_domain_eval")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    decoder = Wav2Vec2Decoder(
        model_name=args.model_name,
        lm_model_path=str(ASSIGNMENT_DIR / args.lm_model_path),
        beam_width=args.beam_width,
        alpha=args.alpha,
        beta=args.beta,
        temperature=1.0,
        device=args.device,
    )

    manifests = [
        ("librispeech_test_other", ASSIGNMENT_DIR / "data" / "librispeech_test_other" / "manifest.csv"),
        ("earnings22_test", ASSIGNMENT_DIR / "data" / "earnings22_test" / "manifest.csv"),
    ]
    methods = [
        ("greedy", "greedy"),
        ("beam", "beam"),
        ("beam_lm", "beam_lm"),
        ("beam_lm_rescore", "beam_lm_rescore"),
    ]

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "method", "wer", "cer", "predictions_csv"],
        )
        writer.writeheader()

        for dataset_name, manifest_path in manifests:
            for method_name, method in methods:
                output_csv = args.output_dir / f"{dataset_name}_{method_name}.csv"
                results = evaluate_manifest(
                    decoder=decoder,
                    manifest_path=manifest_path,
                    method=method,
                    limit=args.limit,
                )
                save_predictions(output_csv, results["predictions"])
                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "method": method_name,
                        "wer": f"{results['wer'] * 100:.2f}",
                        "cer": f"{results['cer'] * 100:.2f}",
                        "predictions_csv": str(output_csv.resolve().relative_to(ASSIGNMENT_DIR.resolve())),
                    }
                )
                print(
                    f"dataset={dataset_name} method={method_name} "
                    f"| WER={results['wer']:.2%} | CER={results['cer']:.2%}"
                )

    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
