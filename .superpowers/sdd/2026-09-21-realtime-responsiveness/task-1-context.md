# Task 1 supplemental context

- Work directly in the current checkout; editing, testing, and committing are authorized.
- Preserve existing dirty changes in `detection_core.py` and `tests/test_detection_core.py`.
- The user approved repairing three baseline failures in Task 1:
  - Update two tests that patch removed `detection_core.DeepFace` to patch the current backend boundary used by `_detect_faces_retinaface`.
  - Apply `min_confidence` consistently to full-frame and tiled detections.
- Baseline: 31 passed, 3 failed.
- Commit only Task 1 files; exclude plans and `.superpowers`.
- Report must contain status, files, commits, exact test commands/results, self-review, and concerns.
