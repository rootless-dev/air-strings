import math
import time

import numpy as np
import pytest

from air_strings.synth import (
    BLOCK_SIZE,
    SAMPLE_RATE,
    VoicePool,
    build_pluck,
    make_excitation,
    triangle_excitation,
)


def render_single(frequency: float, seconds: float, position: float = 0.3, velocity: float = 0.6) -> np.ndarray:
    pool = VoicePool(sample_rate=SAMPLE_RATE, max_block=BLOCK_SIZE)
    pool.trigger(build_pluck(SAMPLE_RATE, 0, frequency, velocity, position, np.random.default_rng(1)))
    return pool.render_offline(int(seconds * SAMPLE_RATE))


def estimate_fundamental(signal: np.ndarray, expected: float) -> float:
    """FFT peak near the expected frequency, zero-padded and refined by parabolic interpolation."""
    windowed = signal * np.hanning(signal.size)
    size = 1 << 21
    magnitude = np.abs(np.fft.rfft(windowed, n=size))
    bin_hz = SAMPLE_RATE / size
    low, high = int(expected * 0.8 / bin_hz), int(expected * 1.2 / bin_hz)
    k = low + int(np.argmax(magnitude[low:high]))
    a, b, c = np.log(magnitude[k - 1 : k + 2] + 1e-20)
    offset = 0.5 * (a - c) / (a - 2 * b + c)
    return (k + offset) * bin_hz


def cents(measured: float, expected: float) -> float:
    return 1200.0 * math.log2(measured / expected)


@pytest.mark.parametrize("frequency", [130.81, 220.0, 440.0, 523.25])
def test_voice_is_in_tune(frequency):
    signal = render_single(frequency, 1.0)
    measured = estimate_fundamental(signal, frequency)
    assert abs(cents(measured, frequency)) < 3.0, f"{measured:.3f} Hz vs {frequency} Hz"


def test_voice_decays():
    signal = render_single(220.0, 3.0)
    window = int(0.25 * SAMPLE_RATE)
    first = np.sqrt(np.mean(signal[:window] ** 2))
    last = np.sqrt(np.mean(signal[-window:] ** 2))
    assert first > 0.05
    assert last < 0.05 * first


def test_higher_notes_decay_faster():
    window = int(0.25 * SAMPLE_RATE)

    def remaining(frequency):
        signal = render_single(frequency, 2.0)
        return np.sqrt(np.mean(signal[-window:] ** 2)) / np.sqrt(np.mean(signal[:window] ** 2))

    assert remaining(523.25) < remaining(130.81)


def test_voice_goes_idle_after_ringing_out():
    pool = VoicePool(sample_rate=SAMPLE_RATE, max_block=BLOCK_SIZE)
    pool.trigger(build_pluck(SAMPLE_RATE, 0, 523.25, 0.5, 0.5, np.random.default_rng(2)))
    pool.render_offline(12 * SAMPLE_RATE)
    assert pool.active_voices == 0


def test_triangle_pluck_in_the_middle_has_no_even_harmonics():
    spectrum = np.abs(np.fft.rfft(triangle_excitation(400, 0.5)))
    fundamental = spectrum[1]
    for harmonic in (2, 4, 6, 8):
        assert spectrum[harmonic] < 1e-9 * fundamental
    assert spectrum[3] > 0.05 * fundamental


def test_excitation_keeps_pluck_position_nulls_and_is_zero_mean():
    excitation = make_excitation(400.0, 0.5, 0.8, np.random.default_rng(3))
    spectrum = np.abs(np.fft.rfft(excitation))
    assert abs(excitation.mean()) < 1e-9
    assert np.max(np.abs(excitation)) == pytest.approx(1.0)
    assert spectrum[2] < 1e-6 * spectrum[1]
    assert spectrum[4] < 1e-6 * spectrum[1]


def test_pluck_near_the_end_is_brighter_than_in_the_middle():
    def high_energy_ratio(position):
        event = build_pluck(SAMPLE_RATE, 0, 120.0, 0.5, position, np.random.default_rng(4))
        spectrum = np.abs(np.fft.rfft(event.excitation)) ** 2
        return spectrum[6:].sum() / spectrum[1:].sum()

    assert high_energy_ratio(0.1) > 2 * high_energy_ratio(0.5)


def test_excitation_starts_at_silence():
    for position in (0.1, 0.3, 0.5, 0.8):
        excitation = make_excitation(300.0, position, 0.7, np.random.default_rng(9))
        assert abs(excitation[0]) < 0.05


def test_replucking_a_string_does_not_click():
    pool = VoicePool(sample_rate=SAMPLE_RATE, max_block=BLOCK_SIZE)
    rng = np.random.default_rng(5)
    pool.trigger(build_pluck(SAMPLE_RATE, 0, 220.0, 0.8, 0.3, rng))
    before = pool.render_offline(SAMPLE_RATE // 2)
    pool.trigger(build_pluck(SAMPLE_RATE, 0, 220.0, 0.8, 0.3, rng))
    after = pool.render_offline(SAMPLE_RATE // 200)
    assert pool.active_voices == 2  # old one fading out, new one ringing
    pool.render_offline(SAMPLE_RATE // 10)
    assert pool.active_voices == 1
    joined = np.concatenate([before[-200:], after[:200]])
    # No step at the retrigger: the old voice fades out and the new one starts at silence.
    assert np.max(np.abs(np.diff(joined[195:206]))) < 0.03


def test_voice_pool_steals_when_full():
    pool = VoicePool(sample_rate=SAMPLE_RATE, voices=3, max_block=BLOCK_SIZE)
    rng = np.random.default_rng(6)
    for string_id in range(5):
        pool.trigger(build_pluck(SAMPLE_RATE, string_id, 220.0 + 20 * string_id, 0.5, 0.4, rng))
    assert pool.active_voices == 3
    assert np.all(np.isfinite(pool.render_offline(BLOCK_SIZE)))


def test_mix_is_soft_limited():
    pool = VoicePool(sample_rate=SAMPLE_RATE, max_block=BLOCK_SIZE)
    rng = np.random.default_rng(7)
    for string_id in range(12):
        pool.trigger(build_pluck(SAMPLE_RATE, string_id, 110.0 * (1 + string_id % 4), 1.0, 0.1, rng))
    assert np.max(np.abs(pool.render_offline(SAMPLE_RATE // 4))) < 1.0


def test_block_with_twelve_voices_renders_faster_than_real_time():
    pool = VoicePool(sample_rate=SAMPLE_RATE, max_block=BLOCK_SIZE)
    rng = np.random.default_rng(8)
    for string_id, midi in enumerate(range(48, 72, 2)):
        frequency = 440.0 * 2 ** ((midi - 69) / 12)
        pool.trigger(build_pluck(SAMPLE_RATE, string_id, frequency, 0.8, 0.3, rng))
    timings = []
    for _ in range(200):
        start = time.perf_counter()
        pool.render(BLOCK_SIZE)
        timings.append(time.perf_counter() - start)
    assert pool.active_voices == 12
    block_duration = BLOCK_SIZE / SAMPLE_RATE
    assert float(np.median(timings)) < 0.25 * block_duration
