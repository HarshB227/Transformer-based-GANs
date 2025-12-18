# ============================
#   FIXED TRAIN GAN (FINAL)
# ============================

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ✅ NEW AMP API (removes FutureWarning)
from torch.amp import autocast, GradScaler


# ---------------------------------
# Fix import path so src/ works
# ---------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                # .../music-gan
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import CFG
from src.utils import (
    set_seed,
    get_device,
    ensure_dir,
    save_checkpoint,
    count_parameters,
)
from src.dataset import MidiGenreDataset
from src.tokenization_small import SmallMidiTokenizer
from src.models.generator import MusicTransformerGenerator
from src.models.discriminator import MusicTransformerDiscriminator


# =========================================================
#  FIXED DATALOADER (READS MAX_FILES FROM CONFIG)
# =========================================================
def build_dataloaders():
    dataset = MidiGenreDataset(
        metadata_csv=CFG.METADATA_CSV,
        midi_dir=CFG.MIDI_ROOT,
        max_len=CFG.MAX_SEQ_LEN,
        max_files=CFG.MAX_FILES,     # controlled from config
    )

    print(f"▶ Final dataset size after limit = {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("❌ No valid MIDI files found. Check your paths!")

    loader = DataLoader(
        dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS if hasattr(CFG, "NUM_WORKERS") else 0,
        drop_last=True,
        pin_memory=(CFG.DEVICE == "cuda"),  # small speedup on GPU
    )
    return loader


# =========================================================
#  BUILD MODELS
# =========================================================
def build_models(vocab_size: int):
    genre_count = len(CFG.GENRES)

    G = MusicTransformerGenerator(
        vocab_size=vocab_size,
        genre_count=genre_count,
        d_model=CFG.D_MODEL,
        n_heads=CFG.N_HEADS,
        num_layers=CFG.GEN_LAYERS,
        latent_dim=CFG.LATENT_DIM,
        max_seq_len=CFG.MAX_SEQ_LEN,
    )

    D = MusicTransformerDiscriminator(
        vocab_size=vocab_size,
        genre_count=genre_count,
        d_model=CFG.D_MODEL,
        n_heads=CFG.N_HEADS,
        num_layers=CFG.DISC_LAYERS,
        max_seq_len=CFG.MAX_SEQ_LEN,
    )

    return G, D


# =========================================================
#  TRAIN LOOP
# =========================================================
def gan_train_loop():
    set_seed(42)
    device = get_device(CFG.DEVICE)
    print(f"\n▶ Using device: {device}")

    tokenizer = SmallMidiTokenizer(max_len=CFG.MAX_SEQ_LEN)
    vocab_size = tokenizer.vocab_size
    print(f"▶ Vocab size: {vocab_size}")

    train_loader = build_dataloaders()

    G, D = build_models(vocab_size)
    G.to(device)
    D.to(device)

    print(f"▶ Generator params: {count_parameters(G):,}")
    print(f"▶ Discriminator params: {count_parameters(D):,}")

    opt_G = torch.optim.Adam(G.parameters(), lr=CFG.LR_GEN, betas=CFG.BETAS)
    opt_D = torch.optim.Adam(D.parameters(), lr=CFG.LR_DISC, betas=CFG.BETAS)

    # Losses
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    # ✅ AMP enabled only on CUDA
    use_amp = (device.type == "cuda")
    scaler_G = GradScaler("cuda", enabled=use_amp)
    scaler_D = GradScaler("cuda", enabled=use_amp)

    ensure_dir(CFG.CHECKPOINT_DIR)
    ensure_dir(CFG.SAMPLES_DIR)

    # ----------------------------------
    # EPOCH LOOP
    # ----------------------------------
    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        G.train()
        D.train()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{CFG.NUM_EPOCHS}",
        )

        for batch_idx, (real_tokens, genre_ids) in pbar:
            real_tokens = real_tokens.long().to(device, non_blocking=use_amp)
            genre_ids = genre_ids.long().to(device, non_blocking=use_amp)
            B, L = real_tokens.size()

            # =============================
            #  TRAIN DISCRIMINATOR
            # =============================
            opt_D.zero_grad(set_to_none=True)

            # ✅ NEW autocast style
            with autocast("cuda", enabled=use_amp):
                real_rf, real_gen_logits = D(real_tokens, genre_ids)
                real_labels = torch.full((B, 1), 0.9, device=device)
                loss_D_real_rf = bce(real_rf, real_labels)
                loss_D_real_gen = ce(real_gen_logits, genre_ids)

                z = torch.randn(B, CFG.LATENT_DIM, device=device)
                fake_logits = G(z, genre_ids, real_tokens)
                fake_tokens = fake_logits.argmax(dim=-1)

                fake_rf, _ = D(fake_tokens, genre_ids)
                fake_labels = torch.zeros(B, 1, device=device)
                loss_D_fake_rf = bce(fake_rf, fake_labels)

                loss_D = loss_D_real_rf + loss_D_real_gen + loss_D_fake_rf

            if use_amp:
                scaler_D.scale(loss_D).backward()
                scaler_D.unscale_(opt_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                scaler_D.step(opt_D)
                scaler_D.update()
            else:
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                opt_D.step()

            # =============================
            #  TRAIN GENERATOR
            # =============================
            opt_G.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                z = torch.randn(B, CFG.LATENT_DIM, device=device)
                fake_logits = G(z, genre_ids, real_tokens)
                fake_tokens = fake_logits.argmax(dim=-1)

                fake_rf_G, fake_gen_logits = D(fake_tokens, genre_ids)
                target_real = torch.full((B, 1), 0.9, device=device)

                loss_G_adv = bce(fake_rf_G, target_real)
                loss_G_genre = ce(fake_gen_logits, genre_ids)
                loss_G_ce = ce(
                    fake_logits.view(-1, fake_logits.size(-1)),
                    real_tokens.view(-1),
                )

                loss_G = (
                    CFG.LAMBDA_ADV * loss_G_adv
                    + CFG.LAMBDA_GENRE * loss_G_genre
                    + CFG.LAMBDA_CE * loss_G_ce
                )

            if use_amp:
                scaler_G.scale(loss_G).backward()
                scaler_G.unscale_(opt_G)
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                scaler_G.step(opt_G)
                scaler_G.update()
            else:
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                opt_G.step()

            if batch_idx % CFG.LOG_INTERVAL == 0:
                pbar.set_postfix({
                    "D_loss": f"{loss_D.item():.4f}",
                    "G_loss": f"{loss_G.item():.4f}",
                })

        # SAVE CHECKPOINT
        ckpt = {
            "epoch": epoch,
            "G_state": G.state_dict(),
            "D_state": D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "vocab_size": vocab_size,
        }
        save_checkpoint(ckpt, os.path.join(CFG.CHECKPOINT_DIR, f"epoch_{epoch}.pt"))

    print("\n✅ Training complete!\n")


if __name__ == "__main__":
    gan_train_loop()
