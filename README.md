# Transformer-based-GANs
Controllable Music Generation Transformer-based GANs
🎵 Music-GAN (Transformer-GAN) — Controllable Multi-Track Symbolic Music Generation
A Transformer-based Generator trained with a GAN-style Discriminator to generate symbolic (MIDI) music with basic controllability (e.g., mood/emotion tags, adding drums/violin, etc.).
This repo operates in the symbolic domain. Audio rendering (MIDI→WAV) is optional.
If your Git pushes keep failing: you are committing huge files (datasets/checkpoints). Don’t. This README includes the correct structure + rules.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
 What this project does
Loads MIDI files (e.g., Lakh MIDI Dataset or your own MIDIs)
Tokenizes MIDI into a compact event sequence
Trains:
MusicTransformerGenerator
MusicTransformerDiscriminator
Generates new sequences → converts back to MIDI
Optional demo script to generate a ~30s sample
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Project Structure 
music-gan/
│
├── README.md
├── requirements.txt              # or pyproject.toml (pick one)
├── .gitignore
│
├── src/
│   └── music_gan/                # your actual python package
│       ├── __init__.py
│       ├── config.py
│       ├── dataset.py
│       ├── tokenization_small.py
│       ├── audio_render.py
│       ├── data_fetch.py
│       ├── clean_metadata.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── generator.py
│       │   └── discriminator.py
│       │
│       └── utils/
│           ├── __init__.py
│           └── *.py
│
├── scripts/                      # runnable entrypoints ONLY
│   ├── train_gan.py
│   ├── run_demo.py
│   └── clean_metadata.py         # optional wrapper
│
├── data/                         # never push raw dataset
│   ├── raw/
│   └── processed/
│
├── checkpoints/                  # don’t push (or use Releases/LFS)
├── outputs/                      # generated midi/wav/plots
└── notebooks/
├─ web/        (only if demo)

------------------------------------------------------------------------------------------------------------------------------------------------------------------
Requirements
Python 3.9+ (recommend 3.10)
OS: Windows / Linux / macOS
Core libs (installed via requirements.txt):
torch
numpy, pandas
pretty_midi
matplotlib
(optional) pyfluidsynth for WAV rendering

------------------------------------------------------------------------------------------------------------------------------------------------------------------
Setup
1) Clone the repo
git clone <YOUR_REPO_URL>
cd music-gan

2) Create and activate a virtual environment
Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

If activation is blocked (PowerShell policy):
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate

3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Data setup (IMPORTANT)
Option A — Use your own MIDI files
Put .mid/.midi files in:
data/raw/

Option B — Lakh MIDI Dataset (recommended)
Download LMD externally and place extracted MIDIs under:
data/raw/

Do NOT commit the dataset to GitHub. It will break your pushes and your repo becomes useless.
Build/clean metadata (if your pipeline uses metadata CSV)
If you have a metadata script like src/clean_metadata.py:
python src/clean_metadata.py


This should produce something like:
data/processed/metadata_4genres.csv
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Training
If your training entry is train_gan.py (or similar), run:
python train_gan.py
Typical outputs:
checkpoint files saved under checkpoints/
logs/metrics printed to terminal
Reality check: Checkpoints can get huge. Don’t push them to GitHub unless you use Releases or proper LFS.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Run the demo (generate MIDI)
If you have run_demo.py:
python run_demo.py
Expected outputs (example):
outputs/generated.mid (or similar)
If your repo disables WAV synthesis, you’ll see a message like:
“MIDI → WAV synthesis disabled … No WAV generated.”
That’s fine — it doesn’t affect generation.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Common problems (and the real fixes)
“ModuleNotFoundError: src”

You’re running from the wrong directory. Run from repo root:
cd music-gan
python run_demo.py
Git push fails / HTTP 500 / RPC failed
You committed multi-GB files (dataset/checkpoints). Fix:
Add them to .gitignore
Remove them from git history (or recreate repo cleanly)
Minimum .gitignore:
.venv/
__pycache__/
data/raw/
checkpoints/
outputs/
*.pt
*.pth
*.ckpt
*.zip
*.tar
*.tar.gz
If you already committed big files, you must remove them from history (BFG or git filter-repo) or start a clean repo.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
What NOT to upload to GitHub
data/raw/ (datasets)
checkpoints/ (training weights)
.venv/
huge generated outputs

Use one of these instead:
GitHub Releases for checkpoint zips
Google Drive link
Hugging Face model repo
Proper Git LFS (only if you truly understand it)
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Citation / Credits
If this project uses Lakh MIDI Dataset / PrettyMIDI / Transformer baselines, cite them properly in your report and add the citations here.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
License
Add a license file if you plan to share publicly (MIT is common). If you don’t add one, default is “all rights reserved”.
