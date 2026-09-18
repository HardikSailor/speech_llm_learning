"""Stage 0 - data.

A speech dataset is just: (waveform, sample_rate, reference_text). Nothing more.
We support a local LibriSpeech-style tree and, as a fallback, a tiny HF dataset so the
pipeline is always runnable.

LibriSpeech layout:
    root/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac
    root/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt   ->  "<utt-id> THE TEXT"
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

from .config import DataConfig


@dataclass
class Utterance:
    uid: str
    path: str
    text: str
    duration: float = -1.0
    # only filled for the in-memory fallback dataset
    array: np.ndarray | None = field(default=None, repr=False)


def _length_csv_for(root: Path) -> Path | None:
    """LibriSpeech on this cluster ships `len_for_bucket/<split>.csv` with lengths in samples.
    Using it means we never open 28k flac headers just to bucket by duration."""
    cand = root.parent / "len_for_bucket" / f"{root.name}.csv"
    return cand if cand.exists() else None


def _durations_from_csv(csv_path: Path, sample_rate: int) -> dict:
    import csv as _csv

    out = {}
    with csv_path.open() as fh:
        for row in _csv.DictReader(fh):
            uid = Path(row["file_path"]).stem
            out[uid] = int(row["length"]) / sample_rate
    return out


def _durations_from_headers(paths: List[str], workers: int = 16) -> List[float]:
    """Fallback: read just the audio header (no decoding), in parallel - it is IO bound."""
    from concurrent.futures import ThreadPoolExecutor

    def one(p: str) -> float:
        info = sf.info(p)
        return info.frames / info.samplerate

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, paths))


def scan_librispeech(root: str, sample_rate: int = 16000) -> List[Utterance]:
    root_p = Path(root)
    if not root_p.exists():
        raise FileNotFoundError(f"data root does not exist: {root}")
    utts: List[Utterance] = []
    for trans in sorted(root_p.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            if not line.strip():
                continue
            uid, text = line.split(" ", 1)
            for ext in (".flac", ".wav"):
                audio = trans.parent / f"{uid}{ext}"
                if audio.exists():
                    utts.append(Utterance(uid=uid, path=str(audio), text=text.strip()))
                    break
    if not utts:
        raise RuntimeError(f"no LibriSpeech-style utterances found under {root}")

    csv_path = _length_csv_for(root_p)
    if csv_path is not None:
        table = _durations_from_csv(csv_path, sample_rate)
        missing = [u for u in utts if u.uid not in table]
        for u in utts:
            u.duration = table.get(u.uid, -1.0)
        if missing:
            for u, d in zip(missing, _durations_from_headers([m.path for m in missing])):
                u.duration = d
    else:
        for u, d in zip(utts, _durations_from_headers([u.path for u in utts])):
            u.duration = d
    return utts


def _load_hf_fallback(name: str) -> List[Utterance]:
    """Tiny HF dataset so the repo runs with no local corpus.
    We decode the bytes ourselves with soundfile: newer `datasets` versions push audio
    decoding onto torchcodec, which we do not need."""
    import io

    from datasets import Audio, load_dataset

    ds = load_dataset(name, "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    for i, ex in enumerate(ds):
        a = ex["audio"]
        raw = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        out.append(
            Utterance(
                uid=ex.get("id", f"utt{i}"),
                path=a.get("path") or "",
                text=ex["text"].strip(),
                duration=len(wav) / sr,
                array=wav,
            )
        )
    return out


class SpeechTextDataset(Dataset):
    """Returns dicts: {"uid", "audio" (float32 np array @16k), "text"}."""

    def __init__(self, cfg: DataConfig, split_ratio: float = 1.0, split: str = "train",
                 limit: int | None = None, budget_minutes: float | None = None, seed: int = 0):
        self.cfg = cfg
        if cfg.root:
            utts = scan_librispeech(cfg.root, cfg.sample_rate)
        else:
            utts = _load_hf_fallback(cfg.hf_fallback)
        self.n_scanned = len(utts)

        # Filter on duration rather than truncating: a truncated waveform still carries the
        # FULL reference text, which teaches the model to hallucinate the unheard tail.
        utts = [u for u in utts if cfg.min_seconds <= u.duration <= cfg.max_seconds]
        self.n_after_filter = len(utts)

        rng = random.Random(seed)
        rng.shuffle(utts)
        n_train = int(len(utts) * split_ratio)
        utts = utts[:n_train] if split == "train" else utts[n_train:]

        if budget_minutes is not None:      # keep only the first N minutes of audio
            kept, total = [], 0.0
            for u in utts:
                if total >= budget_minutes * 60:
                    break
                kept.append(u)
                total += u.duration
            utts = kept
        if limit:
            utts = utts[:limit]
        self.utts = utts

    @property
    def total_hours(self) -> float:
        return sum(u.duration for u in self.utts) / 3600

    @property
    def durations(self) -> List[float]:
        return [u.duration for u in self.utts]

    def __len__(self) -> int:
        return len(self.utts)

    def _read(self, u: Utterance) -> np.ndarray:
        if u.array is not None:
            return u.array
        wav, sr = sf.read(u.path, dtype="float32", always_2d=False)
        if wav.ndim > 1:                      # stereo -> mono
            wav = wav.mean(axis=1)
        if sr != self.cfg.sample_rate:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.cfg.sample_rate)
        return wav

    def __getitem__(self, i: int) -> dict:
        u = self.utts[i]
        wav = self._read(u)
        max_s = self.cfg.max_seconds * self.cfg.sample_rate
        if len(wav) > max_s:                  # keep CPU runs bounded
            wav = wav[: int(max_s)]
        text = u.text.lower() if self.cfg.lowercase else u.text
        return {"uid": u.uid, "audio": wav, "text": text,
                "duration": len(wav) / self.cfg.sample_rate}


def main() -> None:
    ap = argparse.ArgumentParser(description="peek at the dataset")
    ap.add_argument("--root", default=os.environ.get("SPEECH_LLM_DATA", ""))
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    ds = SpeechTextDataset(DataConfig(root=args.root))
    print(f"source      : {args.root or 'HF fallback (' + DataConfig().hf_fallback + ')'}")
    print(f"scanned     : {ds.n_scanned}")
    print(f"after {DataConfig().min_seconds:g}-{DataConfig().max_seconds:g}s filter : "
          f"{ds.n_after_filter}  ({100*ds.n_after_filter/ds.n_scanned:.1f} %)")
    d = sorted(ds.durations)
    if d:
        print(f"duration    : min {d[0]:.1f}s  median {d[len(d)//2]:.1f}s  max {d[-1]:.1f}s  "
              f"total {ds.total_hours:.2f} h")
    durs = []
    for i in range(min(args.n, len(ds))):
        s = ds[i]
        durs.append(s["duration"])
        print("-" * 70)
        print(f"uid       : {s['uid']}")
        print(f"audio     : shape={s['audio'].shape} dtype={s['audio'].dtype} "
              f"{s['duration']:.2f}s  min={s['audio'].min():.3f} max={s['audio'].max():.3f}")
        print(f"text      : {s['text'][:100]}")
        print(f"chars/sec : {len(s['text']) / s['duration']:.1f}")


if __name__ == "__main__":
    main()
