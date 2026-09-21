# Realtime responsiveness design

## Summary

The realtime camera window currently appears to freeze because its event loop runs only after synchronous face detection completes. Each detection cycle performs RetinaFace inference on the full frame and six upscaled tiles, which can occupy the CPU long enough for Windows to mark the OpenCV window as not responding.

This design gives realtime mode a lightweight detection path that prioritizes a responsive 1280x720 display at 24 FPS or better. Uploaded-video processing retains the heavier long-range detector. The existing tracking, recognition, active-speaker, and persistence behavior remains in place.

## Goals

1. Keep the realtime OpenCV window responsive and target at least 24 displayed FPS at 1280x720 on the user's current machine.
2. Preserve face tracking, known-face recognition, unknown-face capture, microphone VAD, and active-speaker overlays.
3. Surface immediate child-process startup failures instead of always reporting success in Streamlit.
4. Allow realtime video recognition to continue when microphone initialization fails.
5. Keep uploaded-video detection quality and behavior unchanged.

## Non-goals

1. Guarantee 24 FPS on every machine or camera backend.
2. Preserve RetinaFace-level long-distance or profile-face detection in realtime mode.
3. Move the native camera display into the Streamlit page.
4. Add GPU inference, multiprocessing, or a general detector framework.
5. Refactor unrelated tracking, database, or transcription code.

## Root cause

`realtime.py` calls `detect_faces_tiled()` synchronously before `cv2.imshow()` and `cv2.waitKey()`. A detection cycle currently includes one full-frame RetinaFace inference and one inference for each tile in the configured 3x2 grid. While those calls run, OpenCV cannot pump native window events. Windows therefore reports the window as not responding even when inference is still progressing.

The Streamlit launcher compounds diagnosis by starting `realtime.py` in a child process and immediately displaying a success message. It does not determine whether that child exited during camera, audio, or model initialization.

## Architecture

### Lightweight detector

`detection_core.py` will expose a public lightweight face-detection function backed by OpenCV's frontal-face Haar cascade. The cascade will be initialized lazily and cached so that realtime frames do not reload its XML model.

The function will:

- accept a BGR frame;
- return detections using the existing `(x, y, width, height, confidence)` contract;
- return an empty list for an empty frame or unavailable cascade;
- contain detector failures and log a warning rather than terminate realtime processing.

Existing tiled RetinaFace behavior will remain available to uploaded-video processing and other current consumers. Current uncommitted work in `detection_core.py` and its tests must be preserved and integrated rather than overwritten.

### Realtime scheduling

`realtime.py` will call the lightweight detector every three captured frames by default instead of calling tiled RetinaFace. The most recent successful detections will be reused on the two intervening frames, allowing `CentroidTracker` to maintain identity while reducing detector cost. If a scheduled detector call fails, its result will be treated as empty and prior boxes will be cleared so stale faces cannot persist indefinitely.

The display loop will continue to call `cv2.imshow()` and `cv2.waitKey(1)` on every captured frame. It will calculate displayed FPS over a rolling one-second window and render it in the window so performance can be checked directly.

ArcFace embedding extraction will retain its existing interval-based schedule in the initial implementation. Recognition scheduling will change only if the manual validation shows that ArcFace prevents the display from reaching 24 FPS after the lightweight detector is active. That follow-up must be based on measured frame timings and covered by an additional test.

### Configuration

`config.py` will define narrowly scoped realtime settings:

- the lightweight detection interval, defaulting to three frames;
- Haar cascade scale and neighbor parameters if tuning is required;
- the rolling FPS measurement window, defaulting to one second.

Defaults will favor smooth local-camera operation. Existing tiled-detection settings will remain unchanged for non-realtime use.

### Streamlit process feedback

`app.py` will retain the separate-process model because OpenCV owns a native window. After launch, Streamlit will wait up to one second for immediate termination while showing a spinner. A nonzero exit during that interval will be reported as a startup failure rather than success. A process still running after one second is considered started; later failures remain visible in its console or startup log and are outside Streamlit's startup check.

The launcher will redirect child output to a uniquely named file under the existing log directory, avoiding a pipe that could block the long-running child. On immediate failure, Streamlit will display at most the final 4 KiB of that file. Once startup succeeds, Streamlit will state that the separate camera window is running. Full live process supervision, log retention management, and a browser-hosted stop control are outside this change.

## Data flow

1. Capture a frame from the selected camera.
2. Pump the OpenCV display event loop every frame.
3. On scheduled detection frames, run lightweight OpenCV face detection.
4. Reuse the latest boxes on intervening frames.
5. Update `CentroidTracker` and process tracked faces.
6. Run landmarks and ArcFace recognition according to their schedules.
7. Apply VAD state, active-speaker selection, database logging, and overlays.
8. Render the frame, FPS, and microphone status.
9. Exit on Q, Escape, camera read failure, or window closure and release all resources.

## Error handling

### Camera and model startup

Camera-open and required model failures remain fatal to the child process. They will produce a clear logged error and a nonzero exit code that the Streamlit launcher can report when failure occurs during its startup check.

### Microphone startup

Microphone or VAD dependency/device failure will be nonfatal. Realtime video detection and recognition will continue with VAD disabled, the overlay will show `MIC: UNAVAILABLE`, and active-speaker logging will remain inactive.

### Runtime failures

- A failed camera frame read ends the loop with a warning.
- A lightweight detector exception logs a warning, clears cached detections, and produces no detections for that cycle.
- Cleanup must tolerate partially initialized camera, VAD, and MediaPipe resources.
- Q and Escape remain supported and must close the window promptly.

## Testing

### Automated tests

Tests will cover:

- lightweight detection returns the existing detection tuple contract;
- invalid or empty frames return no detections;
- the Haar cascade is reused rather than constructed per frame;
- detector failures do not terminate the caller;
- realtime detection scheduling invokes detection only at the configured interval;
- microphone startup failure leaves video processing enabled;
- launcher startup checks distinguish an immediately failed child from a running child.

Tests should mock camera, detector, subprocess, and audio boundaries. They should not require a physical camera or microphone.

### Manual validation

Run realtime mode at 1280x720 on the target machine and verify:

1. The displayed rolling rate reaches at least 24 FPS after warm-up.
2. The camera window remains movable and does not enter Windows' not-responding state.
3. Q and Escape close the process promptly.
4. Known faces are still recognized at normal webcam distance.
5. Unknown-face sampling and database logging still operate.
6. With the microphone unavailable, video continues and shows `MIC: UNAVAILABLE`.
7. An invalid camera index produces an actionable Streamlit startup error.

If 24 FPS is not reached after replacing tiled RetinaFace, timing measurements will identify whether landmark or ArcFace inference exceeds the remaining frame budget. Only that measured bottleneck will receive additional scheduling changes.

## Risks and trade-offs

### Reduced detection range

OpenCV Haar detection is less capable than tiled RetinaFace for distant, rotated, or partially obscured faces. This is an accepted trade-off for the selected maximum-smoothness goal. Uploaded-video mode remains the quality-oriented path.

### Stale boxes between detection frames

Reusing detections can briefly lag fast motion. A short default interval plus centroid tracking limits visible drift, and the interval remains configurable.

### Recognition spikes

ArcFace can still cause occasional frame-time spikes. Rolling FPS instrumentation and elapsed-time scheduling provide a bounded follow-up without prematurely adding threading complexity.

### Platform-specific launcher behavior

The new-console launch path is Windows-specific. Startup-status logic must retain compatible behavior on other operating systems without relying on Windows creation flags.

## Recommended decision

Implement a cached OpenCV detector specifically for realtime mode, schedule it rather than running tiled RetinaFace, keep the native display loop active every frame, degrade gracefully when audio is unavailable, and report immediate child-process failures in Streamlit. This is the smallest change aligned with the approved maximum-smoothness priority and the 24 FPS success criterion.