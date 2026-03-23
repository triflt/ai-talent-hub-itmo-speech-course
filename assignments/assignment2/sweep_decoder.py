import argparse
import csv
from pathlib import Path

from evaluate_decoder import evaluate_manifest, save_predictions, ASSIGNMENT_DIR
from wav2vec2decoder import Wav2Vec2Decoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-base-100h")
    parser.add_argument("--lm-model-path", type=str, default="lm/3-gram.pruned.1e-7.arpa.gz")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)

    subparsers = parser.add_subparsers(dest="mode", required=True)

    beam_parser = subparsers.add_parser("beam")
    beam_parser.add_argument("--beam-widths", type=int, nargs="+", required=True)

    temp_parser = subparsers.add_parser("temperature")
    temp_parser.add_argument("--temperatures", type=float, nargs="+", required=True)

    shallow_parser = subparsers.add_parser("beam_lm")
    shallow_parser.add_argument("--beam-width", type=int, default=10)
    shallow_parser.add_argument("--alphas", type=float, nargs="+", required=True)
    shallow_parser.add_argument("--betas", type=float, nargs="+", required=True)

    rescore_parser = subparsers.add_parser("beam_lm_rescore")
    rescore_parser.add_argument("--beam-width", type=int, default=10)
    rescore_parser.add_argument("--alphas", type=float, nargs="+", required=True)
    rescore_parser.add_argument("--betas", type=float, nargs="+", required=True)

    return parser.parse_args()


def run_single(decoder: Wav2Vec2Decoder, manifest_path: Path, method: str, output_csv: Path, limit: int | None):
    results = evaluate_manifest(
        decoder=decoder,
        manifest_path=manifest_path,
        method=method,
        limit=limit,
    )
    save_predictions(output_csv, results["predictions"])
    return results


def output_path_for_summary(output_csv: Path) -> str:
    output_csv = output_csv.resolve()
    try:
        return str(output_csv.relative_to(ASSIGNMENT_DIR.resolve()))
    except ValueError:
        return str(output_csv)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_lm = args.mode in {"beam_lm", "beam_lm_rescore"}
    lm_path = str(ASSIGNMENT_DIR / args.lm_model_path) if use_lm else None

    initial_beam_width = max(getattr(args, "beam_widths", [3])) if args.mode == "beam" else getattr(args, "beam_width", 3)

    decoder = Wav2Vec2Decoder(
        model_name=args.model_name,
        lm_model_path=lm_path,
        beam_width=initial_beam_width,
        alpha=1.0,
        beta=1.0,
        temperature=1.0,
        device=args.device,
    )

    if args.mode == "beam":
        summary_path = args.output_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["beam_width", "wer", "cer", "predictions_csv"])
            writer.writeheader()
            for beam_width in args.beam_widths:
                decoder.beam_width = beam_width
                output_csv = args.output_dir / f"beam_{beam_width}.csv"
                results = run_single(decoder, args.manifest, "beam", output_csv, args.limit)
                writer.writerow(
                    {
                        "beam_width": beam_width,
                        "wer": f"{results['wer'] * 100:.2f}",
                        "cer": f"{results['cer'] * 100:.2f}",
                        "predictions_csv": output_path_for_summary(output_csv),
                    }
                )
                print(f"beam_width={beam_width} | WER={results['wer']:.2%} | CER={results['cer']:.2%}")
        print(f"Saved: {summary_path}")
        return

    if args.mode == "temperature":
        summary_path = args.output_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["temperature", "wer", "cer", "predictions_csv"])
            writer.writeheader()
            for temperature in args.temperatures:
                decoder.temperature = temperature
                output_csv = args.output_dir / f"greedy_t_{str(temperature).replace('.', '_')}.csv"
                results = run_single(decoder, args.manifest, "greedy", output_csv, args.limit)
                writer.writerow(
                    {
                        "temperature": temperature,
                        "wer": f"{results['wer'] * 100:.2f}",
                        "cer": f"{results['cer'] * 100:.2f}",
                        "predictions_csv": output_path_for_summary(output_csv),
                    }
                )
                print(f"temperature={temperature} | WER={results['wer']:.2%} | CER={results['cer']:.2%}")
        print(f"Saved: {summary_path}")
        return

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["alpha", "beta", "wer", "cer", "predictions_csv"])
        writer.writeheader()
        for alpha in args.alphas:
            for beta in args.betas:
                decoder.alpha = alpha
                decoder.beta = beta
                tag = "beam_lm" if args.mode == "beam_lm" else "beam_rescore"
                output_csv = args.output_dir / f"{tag}_a_{str(alpha).replace('.', '_')}_b_{str(beta).replace('.', '_')}.csv"
                results = run_single(decoder, args.manifest, args.mode, output_csv, args.limit)
                writer.writerow(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "wer": f"{results['wer'] * 100:.2f}",
                        "cer": f"{results['cer'] * 100:.2f}",
                        "predictions_csv": output_path_for_summary(output_csv),
                    }
                )
                print(f"alpha={alpha} beta={beta} | WER={results['wer']:.2%} | CER={results['cer']:.2%}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
