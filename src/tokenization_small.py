def tokens_to_midi_30s(
    tokens,
    out_path: str,
    seconds: float = 30.0,
    emotion: str = "epic",
    add_violin: bool = True,
    add_drums: bool = True,
):
    import pretty_midi
    import numpy as np

    # ==============================
    # ANIME OPENING PRESET
    # ==============================
    emotion = emotion.lower()

    scale = {0, 2, 3, 5, 7, 8, 10}   # minor heroic scale
    tempo = 130
    step_sec = 0.22
    total_steps = int(seconds / step_sec)

    # ==============================
    # Extract pitches
    # ==============================
    pitches = [t for t in tokens if 0 <= t <= 127]
    if len(pitches) < 8:
        pitches = [60, 62, 64, 67, 69, 67, 64, 62] * 16

    def snap(p):
        p = int(np.clip(p, 0, 127))
        pc = p % 12
        if pc in scale:
            return p
        for d in range(1, 7):
            if (pc + d) % 12 in scale:
                return min(127, p + d)
            if (pc - d) % 12 in scale:
                return max(0, p - d)
        return p

    # ==============================
    # MIDI setup
    # ==============================
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    piano = pretty_midi.Instrument(program=0, name="Piano")
    violin = pretty_midi.Instrument(program=40, name="Violin")
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")

    t = 0.0
    idx = 0
    note_len = step_sec * 0.9

    # ==============================
    # STRUCTURED COMPOSITION
    # ==============================
    for s in range(total_steps):
        section = s / total_steps

        raw = pitches[idx % len(pitches)]
        idx += 1

        pitch = snap(raw)

        if section < 0.2:
            pitch -= 12; velocity = 65
        elif section < 0.45:
            velocity = 80
        elif section < 0.75:
            pitch += 12; velocity = 95
        else:
            pitch += 12; velocity = 105

        pitch = int(np.clip(pitch, 48, 88))

        # ---- Piano (always on) ----
        piano.notes.append(pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=t,
            end=t + note_len,
        ))

        # ---- Violin (optional) ----
        if add_violin and section >= 0.3:
            vp = snap(pitch + (7 if s % 2 == 0 else -5))
            violin.notes.append(pretty_midi.Note(
                velocity=velocity - 15,
                pitch=int(np.clip(vp, 55, 95)),
                start=t,
                end=t + note_len,
            ))

        # ---- Drums (optional) ----
        if add_drums and section >= 0.3:
            drums.notes.append(pretty_midi.Note(45, 42, t, t + 0.05))
            if section >= 0.45 and s % 4 == 0:
                drums.notes.append(pretty_midi.Note(95, 36, t, t + 0.08))
            if section >= 0.45 and s % 4 == 2:
                drums.notes.append(pretty_midi.Note(90, 38, t, t + 0.08))

        t += step_sec

    pm.instruments.append(piano)
    if add_violin:
        pm.instruments.append(violin)
    if add_drums:
        pm.instruments.append(drums)

    pm.write(out_path)
    return pm
