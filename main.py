# main.py — CN→EN inference (single sentence only) using your fine-tuned best_model
# Behavior:
#   - Loads ONLY from:  RUN_DIR/best_model/
#   - If not found (or missing config), raises a clear error (no fallback).
#   - Supports single-sentence translate() only.

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerBase

DEFAULT_RUN_DIR = os.environ.get("RUN_DIR", "marian_zh_en_scratch_run")

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class Translator:
    """
    Minimal loader/wrapper:
      - requires RUN_DIR/best_model/
      - single-sentence translate() only
    """
    def __init__(self, run_dir: Optional[str | Path] = None, device: Optional[str] = None):
        self.run_dir = Path(run_dir or DEFAULT_RUN_DIR)
        self.device = device or pick_device()

        # Enforce best_model presence
        self.model_dir = self.run_dir / "best_model"
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"best_model not found at: {self.model_dir}\n"
                f"Make sure you trained and saved best_model under RUN_DIR={self.run_dir}."
            )
        # A quick sanity check for HF-style model folder
        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"config.json not found in {self.model_dir}. The folder does not look like a saved HF model."
            )

        # Load tokenizer + model strictly from best_model
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir)).to(self.device).eval()

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
        """
        Translate ONE Chinese sentence to English.
        Returns (output, info) if return_info=True else just output string.
        """
        text = (text or "").strip()
        if not text:
            if return_info:
                return ("", "Empty input.")
            return ""

        enc = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        gen = self.model.generate(
            **enc,
            num_beams=int(num_beams),
            max_new_tokens=int(max_new_tokens),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            length_penalty=float(length_penalty),
        )
        out = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        info = f"Device: {self.device} • src={self.model_dir} • beams={num_beams} • max_new={max_new_tokens}"
        return (out, info) if return_info else out


if __name__ == "__main__":
    # CLI for quick local testing (single sentence only)
    import argparse, sys
    ap = argparse.ArgumentParser(description="CN→EN Translator CLI (single sentence; best_model only)")
    ap.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR, help="Run dir that MUST contain best_model/.")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--no_repeat_ngram", type=int, default=3)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("text", nargs="*", help="Chinese text (single sentence).")
    args = ap.parse_args()

    try:
        tr = Translator(run_dir=args.run_dir)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    txt = " ".join(args.text).strip()
    if not txt:
        print("Provide TEXT (single sentence). Example:\n  python main.py 我爱学习。", file=sys.stderr)
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
