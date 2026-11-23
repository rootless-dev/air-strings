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

The window (Qt, via PySide6) shows the camera with the strings drawn over it,
the controls on the right and live metrics at the bottom:

```
+------------------------------------------------------------+
| [logo] air-strings                                         |
|  +--------------------------------------+   CAMERA         |
|  |                                      |   [ webcam  v ]  |
|  |       camera, strings and hands      |   [  Clear    ]  |
|  |      (rounded, keeps its aspect)     |   [  Scale    ]  |
|  +--------------------------------------+   gesture hints  |
|============================================================|
| status          display  detection  detect/render ms  ...  |
+------------------------------------------------------------+
```

- **Camera** picks the camera to use (it starts on `--camera`, default 0). The
  new camera opens in the background; if it cannot be opened, the bar says so
  and the current one keeps running. Strings, scale and sound are kept.
- **Clear** removes every string; **Scale** cycles the scale and shows the
  current one.
- The bottom bar shows the displayed frame rate, the detection rate, the hand
  tracker's time per frame, the drawing time per frame, the strings in use, the
  scale and the audio voices sounding.

Camera names come from Qt Multimedia. On macOS, OpenCV numbers cameras in the
order of their AVFoundation unique IDs, so the names are matched to OpenCV's
indices by sorting on that ID. Elsewhere, indices 0 to 4 are probed and shown
as "Camera N".

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

| Key / control      | Action                                     |
|--------------------|--------------------------------------------|
| `s`, **Scale**     | cycle scale: pentatonic, major, chromatic  |
| `c`, **Clear**     | clear all strings                          |
| `q` / Esc / Ctrl+C | quit (closing the window also quits)       |

## Development

```sh
uv run pytest                          # tests (the Qt ones run offscreen)
uv run python scripts/bench_synth.py   # synth timing, 12 voices per 256-frame block
uv run python scripts/bench_render.py  # frame drawing time in a busy 1280x720 scene
```

For bounded smoke runs, `--frames N` quits after N displayed frames and prints
the frame and detection rates; `--headless` runs without any window.

Layout (`src/air_strings/`): `hands.py` (MediaPipe HandLandmarker, One-Euro
smoothing, pinch detection), `pipeline.py` (camera capture and hand
detection on a background thread, newest result wins), `camera.py` (the
camera in use and switching cameras), `geometry.py` (segment crossing), `scales.py`
(scales, length to note), `ruler.py` (ruler geometry), `strings.py` (string
model, vibration), `string_fx.py` (the string's look: ghost strands, wobble,
pulled curve, color gradient), `gestures.py` (draw, grab and pull state
machines), `particles.py` (spark particles), `synth.py` (Karplus-Strong
voices, voice pool, audio stream), `render.py` (OpenCV drawing on the frame),
`engine.py` (the per-frame step shared by the window and the headless loop:
gestures, plucks, sparks, drawing and metrics), `ui/` (the Qt window: video
view, logo, metrics bar, camera names, theme), `app.py` (composition,
lifecycle and the headless loop).
