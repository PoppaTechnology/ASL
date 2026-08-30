import os
import sys
import time
import threading
import queue
import warnings
import logging

import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Force OpenCV's bundled Qt to use a stable platform plugin instead of
# fighting with GNOME/Wayland; xcb (X11) is the most reliable fallback.
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.fonts=false"
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = "sign_language_model.p"
LANDMARKER_PATH = "hand_landmarker.task"

# Camera / performance settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PROCESS_EVERY_N = 2  # run detection+prediction every Nth frame

# Prediction stability / debounce settings
STABILITY_WINDOW = 10
STABILITY_THRESHOLD = 7

# Load model and figure out what features it actually expects
if not os.path.isfile(MODEL_PATH):
    logger.error(f"Model file not found: {MODEL_PATH}")
    sys.exit(1)

model = joblib.load(MODEL_PATH)

if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
    use_z = any(f.startswith("z") for f in feature_names)
    n_features = len(feature_names)
    logger.info(f"Model expects {n_features} features (z-axis included: {use_z})")
else:
    n_features = 42
    use_z = False
    logger.warning("Model has no feature_names_in_; assuming x/y-only 42-feature layout.")

def extract_features(hand_landmarks):
    """Build a flat numpy feature row matching whatever the model was trained on."""
    if use_z:
        coords = [val for lm in hand_landmarks for val in (lm.x, lm.y, lm.z)]
    else:
        coords = [val for lm in hand_landmarks for val in (lm.x, lm.y)]
    return np.array(coords, dtype=np.float32).reshape(1, -1)

# Prediction debouncing: only treat a sign as "real" once it's been
# consistent across most of the recent frames, to avoid flicker.
prediction_history = deque(maxlen=STABILITY_WINDOW)

def get_stable_prediction(new_prediction):
    prediction_history.append(new_prediction)
    if len(prediction_history) < prediction_history.maxlen:
        return None
    most_common = max(set(prediction_history), key=prediction_history.count)
    if prediction_history.count(most_common) >= STABILITY_THRESHOLD:
        return most_common
    return None

# Text-to-speech worker thread (runs in background, never blocks video)
# Queue is capped at size 1 and always holds only the latest word -
# stale/backlogged predictions are dropped rather than spoken late.
speech_queue = queue.Queue(maxsize=1)
_stop_speech = threading.Event()

def queue_speech(text):
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
        except queue.Empty:
            break
    try:
        speech_queue.put_nowait(text)
    except queue.Full:
        pass

def speech_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    while not _stop_speech.is_set():
        try:
            text = speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.warning(f"TTS error: {e}")
    engine.stop()

tts_thread = threading.Thread(target=speech_worker, daemon=True)
tts_thread.start()

# MediaPipe Hand Landmarker — VIDEO mode for sequential frame input
if not os.path.isfile(LANDMARKER_PATH):
    logger.error(f"Landmarker model not found: {LANDMARKER_PATH}")
    sys.exit(1)

base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)

cap = None
detector = None

try:
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam (index 0). Check camera permissions/connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("\n-----------------------------------------------")
    print("  Live ASL Translation Application Running...")
    print("  Click the camera window and tap 'q' to exit safely.")

    last_spoken = ""
    frame_count = 0
    start_time = time.time()

    # Cached state reused on frames we skip processing
    cached_landmarks = None
    cached_prediction = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from webcam; stopping.")
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        run_detection = (frame_count % PROCESS_EVERY_N == 0)

        if run_detection:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            timestamp_ms = int((time.time() - start_time) * 1000)

            try:
                detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                logger.warning(f"Detection failed on this frame: {e}")
                detection_result = None

            if detection_result and detection_result.hand_landmarks:
                cached_landmarks = detection_result.hand_landmarks[0]

                try:
                    input_array = extract_features(cached_landmarks)
                    cached_prediction = model.predict(input_array)[0]
                except Exception as e:
                    logger.warning(f"Prediction failed: {e}")
                    cached_prediction = None

                stable = get_stable_prediction(cached_prediction) if cached_prediction is not None else None
                if stable is not None and stable != last_spoken:
                    print(f"Spoken: {stable}")
                    queue_speech(stable)
                    last_spoken = stable
            else:
                cached_landmarks = None
                cached_prediction = None
                prediction_history.clear()

        # Draw using cached landmarks/prediction on every frame (even skipped ones)
        if cached_landmarks is not None:
            for landmark in cached_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        if cached_prediction is not None:
            cv2.putText(
                frame, f"Sign: {cached_prediction}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,
            )

        cv2.imshow("Live ASL Translation Application", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info(f"User pressed 'q' after {frame_count} frames.")
            break

except Exception as e:
    logger.error(f"Fatal error: {e}")

finally:
    print("\nShutting down system resources safely...")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    if detector is not None:
        detector.close()

    _stop_speech.set()
    queue_speech(None)
    tts_thread.join(timeout=2)

    print("Done. System clean!")
