# Live ASL Translation Application

A real-time computer-vision application that recognizes American Sign Language hand signs from a webcam feed and speaks stable predictions aloud using text-to-speech. The application uses MediaPipe Hand Landmarker to detect one hand, extracts hand-landmark coordinates, passes those features to a pre-trained scikit-learn model, overlays the recognized sign on the camera stream, and announces each newly stabilized prediction.

> **Important:** This project is a sign-recognition prototype. Recognition quality depends on the training data, model classes, lighting, camera quality, hand position, and the user’s signing style. It should not be treated as a complete ASL translation system or as a replacement for human interpreters.

## Features

| Feature | Description |
|---|---|
| Live webcam input | Captures frames from camera device index `0` at a target resolution of `640 × 480`. |
| Hand landmark detection | Uses MediaPipe Hand Landmarker in sequential `VIDEO` mode and processes one hand. |
| Model-driven classification | Loads a serialized scikit-learn-compatible model from `sign_language_model.p`. |
| Flexible feature extraction | Uses either 42 x/y landmark features or 63 x/y/z landmark features, based on the model’s `feature_names_in_` metadata when available. |
| Prediction stabilization | Requires a prediction to remain consistent across a rolling window before treating it as a confirmed sign. |
| Speech output | Runs `pyttsx3` in a background worker so text-to-speech does not block video processing. |
| Visual feedback | Draws detected hand landmarks and displays the current predicted sign in the camera window. |
| Safe shutdown | Releases the webcam, closes OpenCV windows and MediaPipe resources, and stops the speech worker when the user exits. |

## How it works

The application follows this pipeline:

```text
Webcam frame
    ↓
Horizontal flip for mirrored preview
    ↓
MediaPipe hand-landmark detection
    ↓
Feature extraction from landmark coordinates
    ↓
Pre-trained sign classification model
    ↓
Rolling-window stability check
    ↓
On-screen label + text-to-speech output
```

Detection and prediction run every second frame by default (`PROCESS_EVERY_N = 2`). Cached landmarks and predictions are rendered on skipped frames to keep the preview responsive. A prediction history window of 10 processed frames is used, and at least 7 matching predictions are required before a sign is considered stable.

## Requirements

The application requires:

| Requirement | Notes |
|---|---|
| Python | Python 3.10+ is recommended. |
| Webcam | The application opens camera device index `0`. |
| Operating-system audio | `pyttsx3` requires a working text-to-speech backend. |
| GUI/display session | OpenCV displays a live window, so headless environments are not supported without additional changes. |
| Classifier model | A compatible `sign_language_model.p` file must be present in the project root. |
| MediaPipe model | The included `hand_landmarker.task` file must be present in the project root. |

## Installation

Clone or extract the project, then create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\\Scripts\\activate     # Windows PowerShell
```

Install the pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency file includes MediaPipe, OpenCV, NumPy, scikit-learn, joblib, Matplotlib, SciPy, and the `pyttsx3` speech stack. Some entries are platform-specific, including `pywin32`, `pypiwin32`, and `comtypes`; on non-Windows systems, the text-to-speech backend may require additional system packages or configuration.

## Required model files

The source expects two model assets in the same directory as `app.py`:

| File | Purpose | Included in archive? |
|---|---|:---:|
| `hand_landmarker.task` | MediaPipe hand-landmark detector model. | Yes |
| `sign_language_model.p` | Serialized sign classifier loaded with `joblib.load`. | No |

The application exits during startup if either file is missing. Place a compatible classifier at:

```text
sign_language_model.p
```

The classifier must accept the feature layout produced by `extract_features`. If the model exposes `feature_names_in_` and any feature name begins with `z`, the application supplies 63 values per hand landmark (`x`, `y`, and `z`). Otherwise, it assumes a 42-feature x/y-only layout. The model’s output labels are displayed and sent to the speech engine as strings.

## Running the application

From the project root, run:

```bash
python app.py
```

If startup succeeds, a window titled **Live ASL Translation Application** opens. Show one hand to the webcam. The predicted sign appears in the window, and a stable new prediction is spoken aloud.

To exit safely:

1. Click the camera window so it has keyboard focus.
2. Press `q`.

The application also logs camera, detection, prediction, and text-to-speech errors to the console. A failed detection on an individual frame is logged and skipped rather than immediately terminating the processing loop.

## Configuration

The main runtime settings are defined near the top of `app.py`:

| Constant | Default | Purpose |
|---|---:|---|
| `MODEL_PATH` | `sign_language_model.p` | Path to the serialized classifier. |
| `LANDMARKER_PATH` | `hand_landmarker.task` | Path to the MediaPipe landmarker model. |
| `FRAME_WIDTH` | `640` | Requested webcam frame width. |
| `FRAME_HEIGHT` | `480` | Requested webcam frame height. |
| `PROCESS_EVERY_N` | `2` | Runs detection on every Nth frame. |
| `STABILITY_WINDOW` | `10` | Number of recent predictions considered for stabilization. |
| `STABILITY_THRESHOLD` | `7` | Minimum matching predictions required for a stable result. |

For slower hardware, increasing `PROCESS_EVERY_N` may reduce CPU usage. For faster hardware, processing more frames may improve responsiveness. Changing the stability settings involves a trade-off between faster recognition and resistance to flickering predictions.

## Project structure

```text
.
├── app.py                  # Webcam loop, recognition, stabilization, and speech output
├── hand_landmarker.task    # MediaPipe hand-landmark model asset
├── sign_language_model.p   # Required serialized classifier; not included in archive
├── requirements.txt        # Pinned Python dependencies
├── README.md               # Project documentation
└── .gitattributes          # Git attribute configuration
```

## Troubleshooting

### `Model file not found: sign_language_model.p`

The classifier file is not included in the archive. Add a compatible joblib/pickle model named exactly `sign_language_model.p` to the project root, or update `MODEL_PATH` in `app.py`.

### `Landmarker model not found: hand_landmarker.task`

Ensure that the included MediaPipe task file remains in the same directory from which the application is launched, or update `LANDMARKER_PATH` to an absolute or correct relative path.

### `Could not open webcam (index 0)`

Check that a camera is connected, that another application is not using it, and that the operating system has granted camera permission. If the desired camera uses another device index, change `cv2.VideoCapture(0)` in `app.py`.

### OpenCV window or Qt errors

The application sets `QT_QPA_PLATFORM` to `xcb` to prefer an X11-compatible Qt platform. This may help on Linux systems with display compatibility issues, but it requires a graphical session and suitable system libraries. On Windows or macOS, this environment override may need to be adjusted if it conflicts with the local OpenCV installation.

### No speech is produced

Confirm that a text-to-speech engine is installed and that the operating system audio output works. On Linux, `pyttsx3` commonly requires an installed speech-dispatcher or espeak-compatible backend. On Windows, verify that the speech-related dependencies installed successfully and that audio is not muted.

### Predictions flicker or are delayed

The application deliberately waits for a stable prediction before speaking. Adjust `STABILITY_WINDOW` and `STABILITY_THRESHOLD` if necessary. Better lighting, a plain background, a more consistent hand position, and a classifier trained on representative examples may improve recognition.

## Limitations and responsible use

This implementation recognizes individual model classes rather than translating continuous ASL language. It processes one hand only, does not model facial expressions or body pose, and does not provide grammar-aware sentence translation. The classifier and its training labels are not included, so the supported sign vocabulary cannot be determined from the application source alone.

Because the application uses a local webcam and local speech synthesis, camera frames are processed locally by the running process unless the classifier or environment is separately configured to transmit data. Users should still review their operating-system permissions and avoid capturing people without appropriate consent.

## Development suggestions

Useful next steps include adding the missing classifier asset or a training pipeline, documenting the model’s label vocabulary, supporting configurable camera indices, adding a command-line configuration interface, writing automated tests for feature extraction and stabilization, and improving platform-specific installation instructions for text-to-speech backends.

## License

No license file is included in the project archive. Add an appropriate license before redistributing or publishing the application.

## References

[1]: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python "MediaPipe Hand Landmarker for Python"

[2]: https://docs.opencv.org/ "OpenCV documentation"

[3]: https://scikit-learn.org/stable/ "scikit-learn documentation"

[4]: https://pyttsx3.readthedocs.io/ "pyttsx3 documentation"
