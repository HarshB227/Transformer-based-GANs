# ============================================================
# audio_render.py — OPTIONAL MIDI → AUDIO RENDERING
# ============================================================

"""
This module is intentionally OPTIONAL.

The core project operates in the symbolic MIDI domain.
Audio rendering is used only for qualitative demonstration.

To avoid platform-specific dependencies (e.g., FluidSynth),
this function acts as a safe placeholder unless external
rendering is explicitly enabled.
"""

import os
import shutil

def midi_to_wav(
    midi_path: str,
    wav_path: str,
    sf2_path: str = None,
):
    """
    Placeholder MIDI → WAV renderer.

    This function does NOT synthesize audio.
    It exists to keep the pipeline consistent and non-breaking.

    Recommended usage:
    - Render MIDI to WAV externally (DAW, MuseScore, FluidSynth CLI)
    - Save the WAV file to `results/samples/`
    - Analysis scripts operate on WAV only

    Parameters
    ----------
    midi_path : str
        Path to generated MIDI file
    wav_path : str
        Desired WAV output path
    sf2_path : str
        Ignored (kept for API compatibility)
    """

    print("⚠️ audio_render.py: MIDI → WAV synthesis is disabled.")
    print("   This project operates in the symbolic domain.")
    print("   Please render audio externally if required.")

    # Optional: copy a placeholder WAV if it exists
    placeholder = os.path.join(os.path.dirname(wav_path), "placeholder.wav")

    if os.path.isfile(placeholder):
        shutil.copy(placeholder, wav_path)
        print(f"✔ Placeholder audio copied to: {wav_path}")
    else:
        print("ℹ No WAV file generated.")
        print("  (This does NOT affect training or evaluation.)")
