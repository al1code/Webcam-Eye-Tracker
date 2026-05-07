# Resolved Issues

This file tracks runtime issues that were found while bringing the project up and the fixes added to the codebase.

## Fixed in this pass

### 1. App could hang forever when no webcam was available

Symptoms:
- `python eye_tracker.py` appeared to "run forever"
- CPU usage could spike
- No actionable error was shown

Root cause:
- `cv2.VideoCapture(0)` failures were not treated as fatal
- the tracking loop continued with `cap.read()` failures and never surfaced the startup problem

Fix:
- added explicit webcam open validation
- added startup status/error signals
- added a clean user-facing shutdown path when the webcam cannot be opened
- added guarded handling for repeated frame-read failures after startup

### 2. MediaPipe model setup happened too early

Symptoms:
- model initialization happened at import time
- testing and startup diagnostics were harder than necessary

Fix:
- moved model download/loading behind lazy helpers: `ensure_model_file()` and `get_face_landmarker()`

### 3. Final fixation cluster could be dropped from analytics

Symptoms:
- the last fixation in a session might not be included in the final report

Fix:
- `fixation_analysis()` now flushes the final cluster before returning

### 4. No quick non-GUI startup check existed

Fix:
- added `python eye_tracker.py --self-test`
- this validates the model file and webcam access without starting the full overlay session

### 5. 9-point calibration was not truly learning from 9 points

Symptoms:
- calibration could feel inaccurate even after finishing all targets
- gaze tended to drift or compress toward broad screen regions
- the app showed 9 calibration targets but mostly used global percentile ranges

Root cause:
- calibration data was being collapsed into min/max style bounds instead of fitting a mapping from eye ratios to screen coordinates

Fix:
- added per-target sample collection during calibration
- added a polynomial calibration fit from eye ratios to screen coordinates
- kept percentile mapping as a fallback when sample coverage is insufficient
- calibration now auto-finalizes when the 9-point sequence completes

## Verification commands

```bash
py -3.12 -m py_compile eye_tracker.py test_qt.py test_eye_tracker_smoke.py
py -3.12 -m unittest test_eye_tracker_smoke.py
py -3.12 eye_tracker.py --self-test
```

## Notes

- `--self-test` will fail on machines without an available webcam. That is expected and now reported clearly.
- The full app is still interactive by design and stays open until `ESC` is pressed.
