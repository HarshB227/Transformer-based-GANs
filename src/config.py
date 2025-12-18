# src/config.py

class Config:
    # ===== DATA =====
    METADATA_CSV = r"D:\Deep Learning Application\music-gan\data\processed\metadata_4genres.csv"
    MIDI_ROOT    = r"D:\Deep Learning Application\music-gan\data\raw"
    GENRES       = ["rock", "pop", "jazz", "electronic"]

    # ===== SEQ =====
    MAX_SEQ_LEN  = 512
    MAX_FILES    = 2048

    # ===== MODEL (ONLY used for NEW training) =====
    # NOTE: evaluation will auto-detect from checkpoint anyway
    D_MODEL      = 128
    N_HEADS      = 4
    GEN_LAYERS   = 4
    DISC_LAYERS  = 4
    LATENT_DIM   = 64

    # ===== TRAINING =====
    BATCH_SIZE   = 8
    NUM_EPOCHS   = 50
    LR_GEN       = 1e-4
    LR_DISC      = 1e-4
    BETAS        = (0.5, 0.999)
    NUM_WORKERS  = 0
    DEVICE       = "cuda"  # set "cpu" if needed

    # ===== LOSS =====
    LAMBDA_ADV   = 1.0
    LAMBDA_GENRE = 1.0
    LAMBDA_CE    = 5.0

    # ===== OUTPUT =====
    CHECKPOINT_DIR = "results/checkpoints"
    SAMPLES_DIR    = "results/samples"
    FIGURES_DIR    = "results/figures"
    LOG_INTERVAL   = 10


CFG = Config()
