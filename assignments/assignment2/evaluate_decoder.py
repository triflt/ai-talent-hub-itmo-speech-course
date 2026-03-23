import argparse
import csv
from pathlib import Path

import jiwer
import torch
import torchaudio

from wav2vec2decoder import Wav2Vec2Decoder


ASSIGNMENT_DIR = Path(__file__).resolve().parent


def load_manifest(manifest_path: Path, limit: int | None = None):
    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            rows.append(row)
    return rows


def load_audio(audio_path: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform.squeeze(0)


def evaluate_manifest(
        decoder: Wav2Vec2Decoder,
        manifest_path: Path,
        method: str,
        limit: int | None = None,
    ):
    rows = load_manifest(manifest_path, limit=limit)
    predictions = []
    references = []
    hypotheses = []

    for row in rows:
        audio_path = ASSIGNMENT_DIR / row["path"]
        reference = row["text"].strip().lower()
        audio = load_audio(audio_path)
        hypothesis = decoder.decode(audio, method=method).strip().lower()

        predictions.append(
            {
                "path": row["path"],
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": jiwer.wer(reference, hypothesis),
                "cer": jiwer.cer(reference, hypothesis),
            }
        )
        references.append(reference)
        hypotheses.append(hypothesis)

    return {
        "num_samples": len(predictions),
        "wer": jiwer.wer(references, hypotheses),
        "cer": jiwer.cer(references, hypotheses),
        "predictions": predictions,
    }


def save_predictions(path: Path, predictions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "reference", "hypothesis", "wer", "cer"],
        )
        writer.writeheader()
        writer.writerows(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", type=str, default="greedy")
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-base-100h")
    parser.add_argument("--lm-model-path", type=str, default="lm/3-gram.pruned.1e-7.arpa.gz")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    lm_path = None if args.method in {"greedy", "beam"} else str(ASSIGNMENT_DIR / args.lm_model_path)
    decoder = Wav2Vec2Decoder(
        model_name=args.model_name,
        lm_model_path=lm_path,
        beam_width=args.beam_width,
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.temperature,
        device=args.device,
    )

    results = evaluate_manifest(
        decoder=decoder,
        manifest_path=args.manifest,
        method=args.method,
        limit=args.limit,
    )
    print(f"method={args.method}")
    print(f"samples={results['num_samples']}")
    print(f"WER={results['wer']:.2%}")
    print(f"CER={results['cer']:.2%}")

    if args.output_csv is not None:
        save_predictions(args.output_csv, results["predictions"])
        print(f"saved_predictions={args.output_csv}")


if __name__ == "__main__":
    main()
