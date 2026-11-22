"""Procedural plucked-string synthesis (extended Karplus-Strong) and audio output.

Each voice is a Karplus-Strong delay loop with a fractional delay (linear
interpolation) and a one-zero loss filter, merged into a single 3-tap FIR on
the delayed history:

    y[k] = g * (c0 * y[k-N] + c1 * y[k-N-1] + c2 * y[k-N-2])

Every output sample only depends on samples at least N back, so up to N
samples can be computed at once with numpy. A block is processed in chunks of
at most N samples, which keeps the audio callback vectorized and cheap.
"""

from __future__ import annotations

import queue
import sys
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 48_000
BLOCK_SIZE = 256
VOICE_COUNT = 12

LOWEST_FREQUENCY = 30.0
LOOP_FILTER_DELAY = 0.45  # one-zero loss filter weight; 0.5 is classic Karplus-Strong
SILENCE_LEVEL = 1e-4
RETRIGGER_FADE_SECONDS = 0.02
MASTER_GAIN = 0.4


def note_t60(frequency: float) -> float:
    """Decay time to -60 dB: a few seconds, shorter for higher notes."""
    return 4.5 * (130.81 / max(frequency, 1.0)) ** 0.45


def triangle_excitation(length: int, position: float) -> np.ndarray:
    """One period of a triangular displacement peaked at `position` (0..1), peak 1."""
    position = min(0.98, max(0.02, position))
    x = np.arange(length, dtype=np.float64) / length
    rising = x / position
    falling = (1.0 - x) / (1.0 - position)
    return np.minimum(rising, falling)


def make_excitation(
    period: float, position: float, brightness: float, rng: np.random.Generator
) -> np.ndarray:
    """Initial delay-line content for a pluck at `position` with given `brightness` (0..1).

    A triangle peaked at the pluck point plus a little noise that goes through a
    comb filter at the same point (so both share the pluck-position spectral
    nulls), then a circular lowpass whose cutoff rises with brightness. The
    result is zero-mean with unit peak.
    """
    length = max(4, int(round(period)))
    position = min(0.98, max(0.02, position))
    brightness = min(1.0, max(0.0, brightness))

    shape = triangle_excitation(length, position)
    noise = rng.uniform(-1.0, 1.0, length)
    offset = int(round(position * length)) % length
    noise = noise - np.roll(noise, offset)
    noise_peak = np.max(np.abs(noise))
    if noise_peak > 0:
        noise /= noise_peak
    noise_mix = 0.08 + 0.32 * brightness
    signal = (1.0 - noise_mix) * shape + noise_mix * noise

    # The buffer is exactly one period, so filtering in the frequency domain is
    # an exact circular filter: bin k is harmonic k.
    spectrum = np.fft.rfft(signal)
    harmonics = np.arange(spectrum.size, dtype=np.float64)
    cutoff = 3.0 + 40.0 * brightness**1.5
    spectrum *= 1.0 / np.sqrt(1.0 + (harmonics / cutoff) ** 4)
    spectrum[0] = 0.0
    signal = np.fft.irfft(spectrum, n=length)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak
    # The loop plays the buffer from its start: begin at the rising zero crossing
    # closest to silence so the onset has no step (a periodic shift only changes phase).
    rising = np.flatnonzero((signal <= 0.0) & (np.roll(signal, -1) > 0.0))
    if rising.size:
        signal = np.roll(signal, -int(rising[np.argmin(np.abs(signal[rising]))]))
    return signal


@dataclass(frozen=True)
class PluckEvent:
    string_id: int
    frequency: float
    amplitude: float
    t60: float
    excitation: np.ndarray


def build_pluck(
    sample_rate: int,
    string_id: int,
    frequency: float,
    velocity: float,
    position: float,
    rng: np.random.Generator,
) -> PluckEvent:
    """Turn a gesture (velocity 0..1, position 0..1 along the string) into a pluck event."""
    velocity = min(1.0, max(0.0, velocity))
    edge_closeness = abs(2.0 * position - 1.0)  # 0 in the middle, 1 at the ends
    brightness = min(1.0, 0.15 + 0.55 * edge_closeness + 0.3 * velocity)
    excitation = make_excitation(sample_rate / frequency, position, brightness, rng)
    amplitude = 0.2 + 0.8 * velocity**0.8
    return PluckEvent(string_id, frequency, amplitude, note_t60(frequency), excitation)


class KarplusStrongVoice:
    """One plucked-string voice with preallocated buffers (no allocation per block)."""

    def __init__(self, sample_rate: int, max_block: int) -> None:
        self._sample_rate = sample_rate
        self._max_block = max_block
        max_history = int(sample_rate / LOWEST_FREQUENCY) + 3
        self._buffer = np.zeros(max_history + max_block, dtype=np.float64)
        self._scratch = np.zeros(max_block, dtype=np.float64)
        self._ramp = np.arange(1, max_block + 1, dtype=np.float64)
        self.active = False
        self.string_id = -1
        self.started_at = 0
        self._history = 0
        self._delay = 1
        self._coefficients = (0.0, 0.0, 0.0)
        self._amplitude = 0.0
        self._fade_gain = 1.0
        self._fade_step = 0.0

    @property
    def releasing(self) -> bool:
        return self._fade_step > 0.0

    def start(self, event: PluckEvent, started_at: int) -> None:
        period = self._sample_rate / event.frequency
        loss = LOOP_FILTER_DELAY
        delay = period - loss  # interpolated delay; the loss filter adds `loss` samples
        whole = int(delay)
        fraction = delay - whole
        # One period loses a factor g so that the fundamental reaches -60 dB at t60.
        g = 10.0 ** (-3.0 / (event.t60 * event.frequency))
        c0 = (1.0 - loss) * (1.0 - fraction)
        c1 = (1.0 - loss) * fraction + loss * (1.0 - fraction)
        c2 = loss * fraction
        self._coefficients = (g * c0, g * c1, g * c2)
        self._delay = whole
        self._history = whole + 2

        history = self._buffer[: self._history]
        history[:] = 0.0
        excitation = event.excitation[: self._history]
        history[self._history - excitation.size :] = excitation

        self._amplitude = event.amplitude
        self._fade_gain = 1.0
        self._fade_step = 0.0
        self.string_id = event.string_id
        self.started_at = started_at
        self.active = True

    def release(self, fade_samples: int) -> None:
        """Fade out smoothly over `fade_samples` instead of cutting (avoids clicks)."""
        if self.active and not self.releasing:
            self._fade_step = self._fade_gain / max(1, fade_samples)

    def render_add(self, out: np.ndarray, frames: int) -> None:
        """Compute the next `frames` samples and add them into `out`."""
        if not self.active:
            return
        buffer = self._buffer
        scratch = self._scratch
        history = self._history
        delay = self._delay
        k0, k1, k2 = self._coefficients

        done = 0
        while done < frames:
            count = min(delay, frames - done)
            start = history + done
            target = buffer[start : start + count]
            np.multiply(buffer[start - delay : start - delay + count], k0, out=target)
            tmp = scratch[:count]
            np.multiply(buffer[start - delay - 1 : start - delay - 1 + count], k1, out=tmp)
            target += tmp
            np.multiply(buffer[start - delay - 2 : start - delay - 2 + count], k2, out=tmp)
            target += tmp
            done += count

        produced = buffer[history : history + frames]
        tmp = scratch[:frames]
        if self.releasing:
            np.multiply(self._ramp[:frames], -self._fade_step, out=tmp)
            tmp += self._fade_gain
            np.maximum(tmp, 0.0, out=tmp)
            self._fade_gain = float(tmp[-1])
            tmp *= self._amplitude
            tmp *= produced
        else:
            np.multiply(produced, self._amplitude, out=tmp)
        out[:frames] += tmp

        level = float(np.max(np.abs(tmp)))
        # Shift the history window forward (numpy handles the overlap correctly).
        buffer[:history] = buffer[frames : frames + history]

        if (self.releasing and self._fade_gain <= 0.0) or level < SILENCE_LEVEL:
            self.active = False


class VoicePool:
    """Polyphonic pool of Karplus-Strong voices."""

    def __init__(
        self, sample_rate: int = SAMPLE_RATE, voices: int = VOICE_COUNT, max_block: int = 4096
    ) -> None:
        self.sample_rate = sample_rate
        self.max_block = max_block
        self._voices = [KarplusStrongVoice(sample_rate, max_block) for _ in range(voices)]
        self._mix = np.zeros(max_block, dtype=np.float64)
        self._fade_samples = int(RETRIGGER_FADE_SECONDS * sample_rate)
        self._counter = 0

    @property
    def active_voices(self) -> int:
        return sum(1 for v in self._voices if v.active)

    def trigger(self, event: PluckEvent) -> None:
        # Re-plucking a ringing string: let its old vibration fade out quickly
        # while a fresh voice takes over, like a finger touching the string.
        for voice in self._voices:
            if voice.active and voice.string_id == event.string_id:
                voice.release(self._fade_samples)
        voice = self._pick_voice()
        self._counter += 1
        voice.start(event, self._counter)

    def _pick_voice(self) -> KarplusStrongVoice:
        for voice in self._voices:
            if not voice.active:
                return voice
        releasing = [v for v in self._voices if v.releasing]
        candidates = releasing or self._voices
        return min(candidates, key=lambda v: v.started_at)  # steal the oldest

    def render(self, frames: int) -> np.ndarray:
        """Render `frames` mono samples (a view into an internal buffer), soft-limited."""
        if frames > self.max_block:
            raise ValueError(f"block of {frames} frames exceeds max_block={self.max_block}")
        mix = self._mix[:frames]
        mix[:] = 0.0
        for voice in self._voices:
            voice.render_add(mix, frames)
        mix *= MASTER_GAIN
        np.tanh(mix, out=mix)
        return mix

    def render_offline(self, frames: int) -> np.ndarray:
        """Render an arbitrary number of frames (for tests and WAV export)."""
        out = np.empty(frames, dtype=np.float64)
        done = 0
        while done < frames:
            count = min(self.max_block, frames - done)
            out[done : done + count] = self.render(count)
            done += count
        return out


class AudioEngine:
    """Real-time output: plucks go from the video thread to the audio callback via a queue."""

    def __init__(
        self,
        pool: VoicePool,
        block_size: int = BLOCK_SIZE,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._pool = pool
        self._block_size = block_size
        self._events: queue.SimpleQueue[PluckEvent] = queue.SimpleQueue()
        self._rng = rng or np.random.default_rng()
        self._stream = None

    def start(self) -> bool:
        """Open the output stream. Returns False (and stays silent) if no device is usable."""
        try:
            import sounddevice as sd

            self._stream = sd.OutputStream(
                samplerate=self._pool.sample_rate,
                blocksize=self._block_size,
                channels=2,
                dtype="float32",
                latency="low",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as error:  # noqa: BLE001 - any audio failure means "run silent"
            print(f"air-strings: audio output unavailable, running silent: {error}", file=sys.stderr)
            self._stream = None
            return False
        return True

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def pluck(self, string_id: int, frequency: float, velocity: float, position: float) -> None:
        """Called from the video thread; builds the excitation here, off the audio thread."""
        event = build_pluck(
            self._pool.sample_rate, string_id, frequency, velocity, position, self._rng
        )
        self._events.put(event)

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ARG002
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            self._pool.trigger(event)
        done = 0
        while done < frames:
            count = min(self._pool.max_block, frames - done)
            block = self._pool.render(count)
            outdata[done : done + count, 0] = block
            outdata[done : done + count, 1] = block
            done += count
