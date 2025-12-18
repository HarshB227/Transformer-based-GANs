# src/clean_metadata.py

import os
import pandas as pd
import pretty_midi

INPUT_CSV = "data/processed/metadata.csv"
MIDI_ROOT = "data/raw"
OUTPUT_CSV = "data/processed/metadata_clean.csv"

def is_valid_midi(path):
    """Return True if MIDI loads successfully, else False."""
    try:
        _ = pretty_midi.PrettyMIDI(path)
        return True
    except Exception:
        return False


def clean_metadata():
    print("🔍 Loading metadata:", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)

    valid_rows = []
    total = len(df)

    print(f"📦 Total MIDI entries: {total}")

    for i, row in df.iterrows():
        rel_path = row["filepath"]
        midi_path = os.path.join(MIDI_ROOT, rel_path)

        # skip missing files
        if not os.path.exists(midi_path):
            print(f"⚠ Missing file, skipping: {midi_path}")
            continue

        # test if MIDI is OK
        if is_valid_midi(midi_path):
            valid_rows.append(row)
        else:
            print(f"⚠ Corrupt MIDI, skipping: {midi_path}")

    clean_df = pd.DataFrame(valid_rows)
    clean_df.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ Cleaning complete!")
    print(f"🧹 Valid MIDI files: {len(clean_df)}")
    print(f"❌ Corrupt/Invalid: {total - len(clean_df)}")
    print(f"📑 Saved clean metadata to: {OUTPUT_CSV}")


if __name__ == "__main__":
    clean_metadata()
