"""Measure how long the synth takes to render one audio block with 12 active voices.

Run with: uv run python scripts/bench_synth.py
"""

import time

import numpy as np

from air_strings.scales import midi_to_frequency
from air_strings.synth import BLOCK_SIZE, SAMPLE_RATE, VOICE_COUNT, VoicePool, build_pluck


def main() -> None:
    rng = np.random.default_rng(0)
    pool = VoicePool(sample_rate=SAMPLE_RATE, voices=VOICE_COUNT, max_block=BLOCK_SIZE)
    # Spread over C3..A#4; the high notes need the most chunks per block.
    events = [
        build_pluck(SAMPLE_RATE, string_id, midi_to_frequency(48 + 2 * string_id), 0.8, 0.3, rng)
        for string_id in range(VOICE_COUNT)
    ]

    def pluck_all() -> None:
        for event in events:
            pool.trigger(event)
        for _ in range(10):  # let the retrigger fades finish
            pool.render(BLOCK_SIZE)

    pluck_all()
    timings = []
    active = []
    for block in range(2000):
        if block % 100 == 0:  # re-pluck (untimed) so all 12 voices keep ringing
            pluck_all()
        start = time.perf_counter()
        pool.render(BLOCK_SIZE)
        timings.append(time.perf_counter() - start)
        active.append(pool.active_voices)
    timings_ms = np.array(timings) * 1000.0
    block_ms = BLOCK_SIZE / SAMPLE_RATE * 1000.0
    print(f"active voices while timing: min {min(active)}, max {max(active)}")
    print(f"block: {BLOCK_SIZE} frames @ {SAMPLE_RATE} Hz = {block_ms:.2f} ms")
    print(
        f"render time: median {np.median(timings_ms):.3f} ms, "
        f"p99 {np.percentile(timings_ms, 99):.3f} ms, max {timings_ms.max():.3f} ms"
    )
    print(f"median load: {np.median(timings_ms) / block_ms:.1%} of the block duration")


if __name__ == "__main__":
    main()
