# ============================================================
# NOTEBOOK 01 — DATASET EXPLORATION (FINAL FIXED VERSION)
# ============================================================

# ------------------------------------------------------------
# ALWAYS FIX PYTHON PATH FIRST (OR NOTHING WILL WORK)
# ------------------------------------------------------------
import sys, os

PROJECT_ROOT = r"D:\Deep Learning Application\music-gan"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("✔ PROJECT ROOT added:", PROJECT_ROOT)

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import pandas as pd
import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
import random

# ------------------------------------------------------------
# 1. Load metadata
# ------------------------------------------------------------
metadata_path = r"D:\Deep Learning Application\music-gan\data\processed\metadata_4genres.csv"
midi_root     = r"D:\Deep Learning Application\music-gan\data\raw"

print("\nMetadata:", metadata_path)
print("MIDI Root:", midi_root)

df = pd.read_csv(metadata_path)
print("Total rows:", len(df))
print(df.head())

# ------------------------------------------------------------
# 2. Genre distribution
# ------------------------------------------------------------
plt.figure(figsize=(8,4))
df['genre'].value_counts().plot(kind='bar')
plt.title("Genre Distribution")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.show()

# ------------------------------------------------------------
# 3. Pick random MIDI + load with pretty_midi
# ------------------------------------------------------------
idx = random.randint(0, len(df)-1)
row = df.iloc[idx]

midi_path = os.path.join(midi_root, row["filepath"])
print("\nSelected MIDI:", midi_path)

pm = pretty_midi.PrettyMIDI(midi_path)

print("Instruments:", len(pm.instruments))
print("Duration (sec):", pm.get_end_time())
print("Total Notes:", sum(len(inst.notes) for inst in pm.instruments))

# ------------------------------------------------------------
# 4. CLEAN Piano Roll Visualization (cropped)
# ------------------------------------------------------------
fs = 20               # 20 frames/sec
max_seconds = 20      # show first 20 sec only

pr = pm.get_piano_roll(fs=fs)

frames = int(max_seconds * fs)
pr = pr[:, :frames]

plt.figure(figsize=(15,5))
plt.imshow(
    pr,
    aspect="auto",
    origin="lower",
    cmap="inferno",
    interpolation="nearest"
)
plt.title(f"Piano Roll (first {max_seconds} sec)")
plt.xlabel("Time Frames")
plt.ylabel("Pitch")
plt.colorbar(label="Velocity")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. Tokenization test
# ------------------------------------------------------------
from src.tokenization_small import SmallMidiTokenizer

tokenizer = SmallMidiTokenizer(max_seq_len=512)
tokens = tokenizer.midi_to_tokens(midi_path)

print("\nToken count:", len(tokens))
print("First 50 tokens:", tokens[:50])

# ------------------------------------------------------------
# 6. Summary statistics (quick sample)
# ------------------------------------------------------------
lengths = []
notes_count = []

for i in range(50):
    try:
        r = df.iloc[i]
        mpath = os.path.join(midi_root, r["filepath"])
        pm2 = pretty_midi.PrettyMIDI(mpath)

        lengths.append(pm2.get_end_time())
        notes_count.append(sum(len(inst.notes) for inst in pm2.instruments))
    except:
        continue

print("\nAverage duration:", np.mean(lengths))
print("Average notes:", np.mean(notes_count))
