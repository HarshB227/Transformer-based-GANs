# src/dataset.py

import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.tokenization_small import SmallMidiTokenizer as MidiTokenizer


class MidiGenreDataset(Dataset):
    """
    Loads MIDI paths from metadata CSV and returns (tokens, genre_id).

    Expected metadata columns:
      - filepath : path to midi file (absolute or relative)
      - genre_id : integer label

    Defaults:
      - midi_dir = <project_root>/data/raw_midi
    """

    def __init__(
        self,
        metadata_csv,
        midi_dir=None,
        max_len=512,
        max_files=None,
        pad_to_max_len=True,
        retry_on_corrupt=8,
    ):
        self.max_len = int(max_len)
        self.pad_to_max_len = bool(pad_to_max_len)
        self.retry_on_corrupt = int(max(0, retry_on_corrupt))

        self.project_root = Path(__file__).resolve().parents[1]

        # Default midi folder
        if midi_dir is None or str(midi_dir).strip() == "":
            self.midi_dir = (self.project_root / "data" / "raw_midi").resolve()
        else:
            self.midi_dir = Path(midi_dir).resolve()

        # Load metadata CSV
        self.df = pd.read_csv(metadata_csv)

        # Validate required columns
        required_cols = {"filepath", "genre_id"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(
                f"metadata_csv missing columns: {missing}. Required columns: {required_cols}"
            )

        # Build absolute paths
        self.df["abs_path"] = self.df["filepath"].apply(self._to_abs_path)

        # Filter missing files
        exists_mask = self.df["abs_path"].apply(lambda p: Path(p).is_file())
        missing_count = int((~exists_mask).sum())
        if missing_count > 0:
            print(f"⚠ Removing {missing_count} missing MIDI files from metadata.")
            self.df = self.df[exists_mask].reset_index(drop=True)

        # Limit dataset size
        if max_files is not None:
            self.df = self.df.iloc[: int(max_files)].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                "Dataset is empty after filtering.\n"
                f"metadata_csv: {metadata_csv}\n"
                f"midi_dir: {self.midi_dir}\n"
                "Fix: ensure metadata['filepath'] matches actual files on disk."
            )

        print(f"✔ Final dataset size: {len(self.df)}")
        print(f"✔ MIDI root: {self.midi_dir}")

        # Tokenizer
        self.tokenizer = MidiTokenizer(max_seq_len=self.max_len)

    def _to_abs_path(self, fp):
        """Convert metadata filepath to absolute path inside midi_dir."""
        if not isinstance(fp, str):
            fp = str(fp)

        fp_norm = fp.replace("\\", "/").lstrip("./")
        p = Path(fp_norm)

        # already absolute -> keep
        if p.is_absolute():
            return str(p)

        # remove common prefixes if metadata includes them
        prefixes = [
            "data/raw_midi/",
            "data/raw/",
            "raw_midi/",
            "raw/",
            "midi/",
        ]
        for pre in prefixes:
            if fp_norm.startswith(pre):
                fp_norm = fp_norm[len(pre):]
                break

        return str((self.midi_dir / fp_norm).resolve())

    def __len__(self):
        return len(self.df)

    def _pad_or_truncate(self, tokens):
        if len(tokens) >= self.max_len:
            return tokens[: self.max_len]
        if self.pad_to_max_len:
            return tokens + [0] * (self.max_len - len(tokens))
        return tokens

    def __getitem__(self, idx):
        n = len(self.df)
        idx = int(idx) % n

        last_err = None
        for k in range(self.retry_on_corrupt + 1):
            row = self.df.iloc[(idx + k) % n]
            midi_path = row["abs_path"]

            try:
                genre_id = int(row["genre_id"])
            except Exception:
                genre_id = 0

            try:
                tokens = self.tokenizer.midi_to_tokens(midi_path)
                tokens = self._pad_or_truncate(tokens)

                tokens = torch.tensor(tokens, dtype=torch.long)
                genre_id = torch.tensor(genre_id, dtype=torch.long)
                return tokens, genre_id

            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"Failed to load MIDI after retries. Last error: {last_err}")
