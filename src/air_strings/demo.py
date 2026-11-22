"""Offline demo: render a strummed chord and an arpeggio to a WAV file (no webcam needed)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from air_strings.scales import midi_to_frequency
from air_strings.synth import SAMPLE_RATE, VoicePool, build_pluck

# (time in seconds, string id, MIDI note, velocity, pluck position)
_STRUM_DOWN = [(0.0 + 0.03 * i, i, m, 0.8, 0.3) for i, m in enumerate((48, 55, 60, 64, 67, 72))]
_ARPEGGIO = [
    (2.4 + 0.22 * i, i % 6, m, 0.55, 0.5)
    for i, m in enumerate((48, 52, 55, 60, 64, 67, 69, 67, 64, 60, 57, 55))
]
_STRUM_UP = [(5.6 + 0.025 * i, i, m, 1.0, 0.12) for i, m in enumerate((69, 64, 60, 57, 52, 45))]
DEMO_EVENTS = _STRUM_DOWN + _ARPEGGIO + _STRUM_UP
DEMO_SECONDS = 9.0


def render_demo(path: Path, sample_rate: int = SAMPLE_RATE, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    pool = VoicePool(sample_rate=sample_rate, max_block=256)
    total = int(DEMO_SECONDS * sample_rate)
    events = sorted(DEMO_EVENTS)
    out = np.zeros(total, dtype=np.float64)
    position = 0
    next_event = 0
    while position < total:
        # Trigger every event due before the next block, like the audio callback does.
        while next_event < len(events) and events[next_event][0] * sample_rate <= position:
            _, string_id, midi, velocity, pluck_at = events[next_event]
            pool.trigger(
                build_pluck(sample_rate, string_id, midi_to_frequency(midi), velocity, pluck_at, rng)
            )
            next_event += 1
        count = min(256, total - position)
        out[position : position + count] = pool.render(count)
        position += count

    pcm = (np.clip(out, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return path
