# air-strings

A computer-vision air instrument. Draw guitar strings in the air with your hands
in front of the webcam, then pluck them with your index finger. Plucked strings
vibrate on screen and sound through a procedural Karplus-Strong synthesizer (no
samples).

## Run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv run air-strings                     # or: uv run python -m air_strings
uv run air-strings --camera 1 --width 1280 --height 720
```

The hand model (`models/hand_landmarker.task`) is downloaded automatically on
the first run. On macOS, allow camera access for your terminal app
(System Settings > Privacy & Security > Camera).

No webcam at hand? Render an audio demo (a strummed chord and an arpeggio):

```sh
uv run air-strings --render-demo demo.wav
```

## Playing

- **Draw a string**: pinch (thumb + index) with both hands, pull them apart, and
  release either pinch. The string stays where you left it. Up to 6 strings; a
  7th replaces the oldest.
- **Pitch**: like a real string, shorter is higher. The length is mapped over
  two octaves (C3 to C5) and snapped to the active scale. A ruler runs above
  every string, end to end; it lights up while the string is being stretched,
  next to the note it plays now.
- **Resize**: pinch one end of a string with one hand and drag it; the other
  end stays put and the note follows the length. A one-hand pinch next to an
  endpoint always grabs it, so start a new string away from existing endpoints.
- **Pluck**: with an open hand, push your index fingertip through a string. It
  catches on the fingertip and bends; once bent far enough (about a tenth of
  its length) it slips off and rings, louder the further it was pulled.
  Plucking near the ends sounds brighter, in the middle rounder. A fast flick
  plucks straight away; sweep across several strings to strum. Backing the
  finger out across the string lets it go silently; sliding off one of its
  ends lets it go with a soft pluck.

| Key        | Action                                            |
|------------|---------------------------------------------------|
| `s`        | cycle scale: pentatonic, major, chromatic         |
| `c`        | clear all strings                                 |
| `q` / Esc  | quit                                              |

## Development

```sh
uv run pytest                          # tests
uv run python scripts/bench_synth.py   # synth timing, 12 voices per 256-frame block
uv run python scripts/bench_render.py  # frame drawing time in a busy 1280x720 scene
```

Layout (`src/air_strings/`): `hands.py` (MediaPipe HandLandmarker, One-Euro
smoothing, pinch detection), `geometry.py` (segment crossing), `scales.py`
(scales, length to note), `ruler.py` (ruler geometry), `strings.py` (string
model, vibration), `string_fx.py` (the string's look: ghost strands, wobble,
pulled curve, color gradient), `gestures.py` (draw, grab and pull state
machines), `particles.py` (spark particles), `synth.py` (Karplus-Strong
voices, voice pool, audio stream), `render.py` (OpenCV drawing), `app.py`
(composition and main loop).
