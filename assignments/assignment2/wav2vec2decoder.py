import math
import os
from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def _log_add(a: float, b: float) -> float:
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-100h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0,
            temperature=1.0,
            device="auto",
        ):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None
        self.non_ctc_token_ids = {
            idx for idx, token in self.vocab.items()
            if token in {"<s>", "</s>", "<unk>"}
        }
        self.decoding_token_ids = [
            idx for idx in sorted(self.vocab)
            if idx != self.blank_token_id and idx not in self.non_ctc_token_ids
        ]

    def _ids_to_text(self, token_ids: List[int]) -> str:
        text = ''.join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, ' ').strip().lower()

    def _acoustic_score(self, p_blank: float, p_non_blank: float) -> float:
        return _log_add(p_blank, p_non_blank)

    def _lm_score(self, text: str, eos: bool) -> float:
        if not self.lm_model or not text:
            return 0.0
        # kenlm uses log10, torch uses ln
        return self.lm_model.score(text, bos=True, eos=eos) * math.log(10.0)

    def _completed_text(self, token_ids: Tuple[int, ...]) -> Tuple[str, int]:
        if not token_ids:
            return "", 0
        text = self._ids_to_text(list(token_ids))
        if token_ids[-1] != self.word_delimiter:
            words = text.split()[:-1]
            return " ".join(words), len(words)
        words = text.split()
        return text, len(words)

    def _final_text_score(self, token_ids: Tuple[int, ...]) -> Tuple[str, int, float]:
        text = self._ids_to_text(list(token_ids))
        if not text:
            return "", 0, 0.0
        word_count = len(text.split())
        return text, word_count, self._lm_score(text, eos=True)

    def _beam_rank_score(
            self,
            token_ids: Tuple[int, ...],
            acoustic_score: float,
            use_lm: bool,
        ) -> float:
        if not use_lm or not self.lm_model:
            return acoustic_score
        completed_text, completed_words = self._completed_text(token_ids)
        return acoustic_score + self.alpha * self._lm_score(completed_text, eos=False) + self.beta * completed_words

    def _final_rank_score(
            self,
            token_ids: Tuple[int, ...],
            acoustic_score: float,
            use_lm: bool,
        ) -> float:
        if not use_lm or not self.lm_model:
            return acoustic_score
        _, word_count, lm_score = self._final_text_score(token_ids)
        return acoustic_score + self.alpha * lm_score + self.beta * word_count

    def _prefix_beam_search(
            self,
            logits: torch.Tensor,
            use_lm: bool = False,
        ) -> List[Tuple[List[int], float]]:
        log_probs = torch.log_softmax(logits, dim=-1).cpu()
        neg_inf = float('-inf')
        beams = {(): (0.0, neg_inf)}

        for frame in log_probs:
            next_beams = {}
            blank_log_prob = frame[self.blank_token_id].item()

            for prefix, (p_blank, p_non_blank) in beams.items():
                prefix_score = self._acoustic_score(p_blank, p_non_blank)

                next_p_blank, next_p_non_blank = next_beams.get(prefix, (neg_inf, neg_inf))
                next_p_blank = _log_add(next_p_blank, prefix_score + blank_log_prob)
                next_beams[prefix] = (next_p_blank, next_p_non_blank)

                last_token = prefix[-1] if prefix else None
                for token_id in self.decoding_token_ids:
                    token_log_prob = frame[token_id].item()
                    if token_id == last_token:
                        stay_p_blank, stay_p_non_blank = next_beams.get(prefix, (neg_inf, neg_inf))
                        stay_p_non_blank = _log_add(stay_p_non_blank, p_non_blank + token_log_prob)
                        next_beams[prefix] = (stay_p_blank, stay_p_non_blank)

                        extended_prefix = prefix + (token_id,)
                        ext_p_blank, ext_p_non_blank = next_beams.get(extended_prefix, (neg_inf, neg_inf))
                        ext_p_non_blank = _log_add(ext_p_non_blank, p_blank + token_log_prob)
                        next_beams[extended_prefix] = (ext_p_blank, ext_p_non_blank)
                    else:
                        extended_prefix = prefix + (token_id,)
                        ext_p_blank, ext_p_non_blank = next_beams.get(extended_prefix, (neg_inf, neg_inf))
                        ext_p_non_blank = _log_add(ext_p_non_blank, prefix_score + token_log_prob)
                        next_beams[extended_prefix] = (ext_p_blank, ext_p_non_blank)

            ranked = sorted(
                next_beams.items(),
                key=lambda item: self._beam_rank_score(
                    item[0],
                    self._acoustic_score(item[1][0], item[1][1]),
                    use_lm=use_lm,
                ),
                reverse=True,
            )
            beams = dict(ranked[:self.beam_width])

        ranked = sorted(
            beams.items(),
            key=lambda item: self._final_rank_score(
                item[0],
                self._acoustic_score(item[1][0], item[1][1]),
                use_lm=use_lm,
            ),
            reverse=True,
        )
        return [
            (list(prefix), self._acoustic_score(p_blank, p_non_blank))
            for prefix, (p_blank, p_non_blank) in ranked[:self.beam_width]
        ]

    def greedy_decode(self, logits: torch.Tensor) -> str:
        best_path = torch.argmax(logits, dim=-1).tolist()
        collapsed = []
        prev_token = None
        for token_id in best_path:
            if token_id == prev_token:
                continue
            prev_token = token_id
            if token_id == self.blank_token_id or token_id in self.non_ctc_token_ids:
                continue
            collapsed.append(token_id)
        return self._ids_to_text(collapsed)

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        beams = self._prefix_beam_search(logits, use_lm=False)
        if return_beams:
            return beams
        return self._ids_to_text(beams[0][0]) if beams else ""

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        beams = self._prefix_beam_search(logits, use_lm=True)
        return self._ids_to_text(beams[0][0]) if beams else ""

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        best_text = ""
        best_score = float('-inf')

        for token_ids, acoustic_score in beams:
            text = self._ids_to_text(token_ids)
            word_count = len(text.split()) if text else 0
            total_score = acoustic_score + self.alpha * self._lm_score(text, eos=True) + self.beta * word_count
            if total_score > best_score:
                best_score = total_score
                best_text = text

        return best_text

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        input_values = inputs.input_values.to(self.device)
        with torch.no_grad():
            logits = self.model(input_values).logits[0].cpu()

        logits = logits / self.temperature

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        raise ValueError(f"Unknown method: {method}")

def test(decoder: Wav2Vec2Decoder, audio_path: str, reference: str) -> None:
    import jiwer

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    print("=" * 60)
    print(f"REF : {reference}")

    for method in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        try:
            hyp = decoder.decode(audio_input, method=method)
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue
        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        print(f"  [{method}] {hyp}")
        print(f"           WER={wer:.2%}  CER={cer:.2%}")


if __name__ == "__main__":
    test_samples = [
        ("examples/sample1.wav", "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance"),
        ("examples/sample2.wav", "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months"),
        ("examples/sample3.wav", "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin"),
        ("examples/sample4.wav", "it was a tune they had all heard hundreds of times so there was no difficulty in turning out a passable imitation of it to the improvised strains of i didn't want to do it the prisoner strode forth to freedom"),
        ("examples/sample5.wav", "marguerite tired out with this long confession threw herself back on the sofa and to stifle a slight cough put up her handkerchief to her lips and from that to her eyes"),
        ("examples/sample6.wav", "at this time all participants are in a listen only mode"),
        ("examples/sample7.wav", "the increase was mainly attributable to the net increase in the average size of our fleets"),
        ("examples/sample8.wav", "operating surplus is a non cap financial measure which is defined as fully in our press release"),
    ]

    device = os.getenv("ASR_DEVICE", "auto")
    model_name = os.getenv("ASR_MODEL_NAME", "facebook/wav2vec2-base-100h")
    decoder = Wav2Vec2Decoder(
        model_name=model_name,
        lm_model_path=None,
        device=device,
    )

    for audio_path, reference in test_samples:
        test(decoder, audio_path, reference)
