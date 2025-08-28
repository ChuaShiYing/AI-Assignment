# main.py — Core inference utilities for CN→EN translation (Marian)
# Loading priority:
#   1) RUN_DIR/best_model/ (fine-tuned weights saved by your notebook)
#   2) HF_MODEL env var (a Hugging Face repo id, e.g., "yourname/marian-zh-en-ft")
#   3) Default base model: "Helsinki-NLP/opus-mt-zh-en"
#
# GPU is used automatically if available (CUDA/MPS). Falls back to CPU.

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerBase

DEFAULT_MODEL_NAME = os.environ.get("HF_MODEL", "Helsinki-NLP/opus-mt-zh-en")
DEFAULT_RUN_DIR = os.environ.get("RUN_DIR", "marian_zh_en_scratch_run")

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class Translator:
    """
    Simple loader/wrapper for Marian-like seq2seq models.
    """

    def __init__(
        self,
        run_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
        prefer_best: bool = True,
    ):
        self.run_dir = Path(run_dir or DEFAULT_RUN_DIR)
        self.device = device or pick_device()

        # Choose source to load
        best_dir = self.run_dir / "best_model"
        if prefer_best and best_dir.exists():
            self.model_src: str | Path = str(best_dir)
        elif os.environ.get("HF_MODEL"):
            self.model_src = os.environ["HF_MODEL"]
        else:
            self.model_src = DEFAULT_MODEL_NAME

        # Load tokenizer + model
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self.model_src)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_src).to(self.device).eval()

    @torch.no_grad()
    def translate(
        self,
        text: str,
        num_beams: int = 4,
        max_new_tokens: int = 128,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.0,
        return_info: bool = False,
    ) -> Tuple[str, Optional[str]]:
        """Translate a single sentence from Chinese to English."""
        text = (text or "").strip()
        if not text:
            return ("", "Empty input.") if return_info else ""

        enc = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        gen = self.model.generate(
            **enc,
            num_beams=int(num_beams),
            max_new_tokens=int(max_new_tokens),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            length_penalty=float(length_penalty),
        )
        out = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        info = f"Device: {self.device} • src={self.model_src} • beams={num_beams} • max_new={max_new_tokens}"
        return (out, info) if return_info else out

    @torch.no_grad()
    def translate_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
        num_beams: int = 4,
        max_new_tokens: int = 128,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.0,
    ) -> List[str]:
        """Translate many sentences (list of CN strings) to EN."""
        outs: List[str] = []
        for i in range(0, len(texts), batch_size):
            batch = [t.strip() for t in texts[i:i + batch_size] if t and t.strip()]
            if not batch:
                continue
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(self.device)
            gen = self.model.generate(
                **enc,
                num_beams=int(num_beams),
                max_new_tokens=int(max_new_tokens),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
                length_penalty=float(length_penalty),
            )
            outs.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return outs


if __name__ == "__main__":
    # Small CLI for local testing
    import argparse, sys
    ap = argparse.ArgumentParser(description="CN→EN Translator CLI (Marian)")
    ap.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR, help="Run dir that may contain best_model/.")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--no_repeat_ngram", type=int, default=3)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--file", type=str, default=None, help="TXT with one sentence per line.")
    ap.add_argument("text", nargs="*", help="Chinese text (if --file not used).")
    args = ap.parse_args()

    tr = Translator(run_dir=args.run_dir)
    if args.file:
        lines = [ln.strip() for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln.strip()]
        outs = tr.translate_batch(
            lines,
            batch_size=16,
            num_beams=args.beams,
            max_new_tokens=args.max_new,
            no_repeat_ngram_size=args.no_repeat_ngram,
            length_penalty=args.length_penalty,
        )
        for o in outs:
            print(o)
    else:
        txt = " ".join(args.text).strip()
        if not txt:
            print("Provide TEXT or --file path.", file=sys.stderr)
            sys.exit(1)
        out, info = tr.translate(
            txt,
            num_beams=args.beams,
            max_new_tokens=args.max_new,
            no_repeat_ngram_size=args.no_repeat_ngram,
            length_penalty=args.length_penalty,
            return_info=True,
        )
        print(out)
        print(f"[{info}]")
