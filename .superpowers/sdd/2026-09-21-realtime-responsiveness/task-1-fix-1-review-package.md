# Task 1 Fix 1 Review Package

## HEAD
fa0126be2749818709f468e3a49e9f5c38e96a0e

## Commit list
fa0126b (HEAD -> main) fix(task1): use module-like DeepFace doubles in tests; portable Haar cascade path

## Stat
```text
 .../task-1-report.md                               | 13 +++++++++
 detection_core.py                                  |  7 ++++-
 tests/test_detection_core.py                       | 34 +++++++++++++++-------
 3 files changed, 43 insertions(+), 11 deletions(-)
```

## Changed tracked files
```text
.superpowers/sdd/2026-09-21-realtime-responsiveness/task-1-report.md
detection_core.py
tests/test_detection_core.py
```

## Full diff (`git diff -U10 
34c1e84f470e2add376fdb89c6eaff6195b44a1d
..HEAD`)
```diff
diff --git a/.superpowers/sdd/2026-09-21-realtime-responsiveness/task-1-report.md b/.superpowers/sdd/2026-09-21-realtime-responsiveness/task-1-report.md
index c26e29e..10e6c69 100644
--- a/.superpowers/sdd/2026-09-21-realtime-responsiveness/task-1-report.md
+++ b/.superpowers/sdd/2026-09-21-realtime-responsiveness/task-1-report.md
@@ -6,20 +6,33 @@ Commits:
 Files changed:
 - config.py
 - detection_core.py
 - tests/test_config.py (appended test already present)
 - tests/test_detection_core.py (updated and added OpenCV tests)
 
 Test summary:
 - Command: `python -m pytest tests/test_config.py tests/test_detection_core.py -q -v`
 - Result: 25 passed in 0.16s
 
+Fix details performed:
+
+- Tests: Replaced Mock-based module injection of `deepface` with `types.ModuleType` module-like doubles in [tests/test_detection_core.py](tests/test_detection_core.py#L96-L114) and [tests/test_detection_core.py](tests/test_detection_core.py#L129-L148) to more accurately simulate import-time module behavior. This prevents importing the real `deepface` package while allowing `DeepFace.extract_faces` to be called normally.
+- Code: Made Haar cascade path construction portable in `detection_core.py` by using `os.path.join` and safe attribute access (`getattr(cv2, "data", None)` / `getattr(..., "haarcascades", "")`) so tests can override `cv2.data.haarcascades` when monkeypatching `cv2`. Change location: [detection_core.py](detection_core.py#L24-L32).
+
+Test run (exact):
+
+```
+python -m pytest tests/test_config.py tests/test_detection_core.py -q -q
+
+25 passed
+```
+
 Self-review:
 - Implemented `REALTIME_*` configuration constants in `config.py` as requested.
 - Added a public `detect_faces_opencv()` that uses a cached OpenCV `CascadeClassifier` with a lock.
 - Kept the existing private `_detect_faces_opencv()` as a fallback; `detect_faces_opencv()` mirrors its behavior.
 - Updated tests to avoid importing the real `deepface` package by injecting a fake module and to assert cascade caching without relying on system `cv2` internals.
 - Preserved existing detection logic and applied `min_confidence` consistently to both full-frame and tiled detections per supplemental context.
 
 Concerns:
 - Tests mock `cv2` and `deepface` to avoid heavy native dependencies; behavior on real hardware/OpenCV builds should be verified in an integration environment.
 - Default confidence value for OpenCV detections is set to 0.7 arbitrarily; this may need tuning.
diff --git a/detection_core.py b/detection_core.py
index ff51f4b..bfea6da 100644
--- a/detection_core.py
+++ b/detection_core.py
@@ -3,39 +3,44 @@
 The geometry and NMS functions are dependency-light so they can be tested
 without loading the RetinaFace model.  Model inference is lazy and failures
 in one tile are isolated from the remaining tiles.
 """
 
 import logging
 import threading
 from typing import List, Tuple
 
 import cv2
+import os
 import numpy as np
 
 
 logger = logging.getLogger(__name__)
 
 Detection = Tuple[float, float, float, float, float]
 
 
 # Lightweight cached OpenCV cascade for fallback/realtime detection
 _OPENCV_CASCADE = None
 _OPENCV_CASCADE_LOCK = threading.Lock()
 
 
 def _get_opencv_cascade():
     global _OPENCV_CASCADE
     if _OPENCV_CASCADE is None:
         with _OPENCV_CASCADE_LOCK:
             if _OPENCV_CASCADE is None:
-                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
+                # Build the cascade path portably so tests can override cv2.data.haarcascades
+                cascade_filename = 'haarcascade_frontalface_default.xml'
+                base = getattr(cv2, "data", None)
+                haar_dir = getattr(base, "haarcascades", "") if base is not None else ""
+                cascade_path = os.path.join(haar_dir, cascade_filename)
                 _OPENCV_CASCADE = cv2.CascadeClassifier(cascade_path)
     return _OPENCV_CASCADE
 
 
 def _starts(length: int, tile_length: int, count: int) -> List[int]:
     if count <= 1:
         return [0]
     if tile_length >= length:
         return [0] * count
 
diff --git a/tests/test_detection_core.py b/tests/test_detection_core.py
index 031f2eb..7ecd8ac 100644
--- a/tests/test_detection_core.py
+++ b/tests/test_detection_core.py
@@ -92,27 +92,34 @@ def test_detect_faces_filters_low_confidence_detections():
         {
             "facial_area": {"x": 4, "y": 6, "w": 12, "h": 14},
             "confidence": 0.3,  # Below 0.5 threshold
         },
         {
             "facial_area": {"x": 20, "y": 20, "w": 15, "h": 15},
             "confidence": 0.9,  # Above 0.5 threshold
         }
     ]
     
-    # Patch DeepFace.extract_faces to the current backend boundary used by _detect_faces_retinaface
-    # Inject a fake deepface module to avoid importing the real dependency
-    fake_deepface = Mock()
-    fake_deepface.DeepFace = Mock()
-    fake_deepface.DeepFace.extract_faces = Mock(return_value=fake_response)
+    # Inject a fake deepface module using types.ModuleType to avoid importing
+    # the real dependency. This more closely models the module import boundary.
+    import types
     import sys
-    with patch.dict(sys.modules, {"deepface": fake_deepface}):
+
+    module = types.ModuleType("deepface")
+
+    class DeepFace:
+        @staticmethod
+        def extract_faces(*_args, **_kwargs):
+            return fake_response
+
+    module.DeepFace = DeepFace
+    with patch.dict(sys.modules, {"deepface": module}):
         detections = _detect_faces_retinaface(tile)
     
     # Only the high-confidence detection should be returned
     assert len(detections) == 1
     assert detections[0][4] == 0.9
 
 
 def test_detect_faces_falls_back_to_opencv_when_retinaface_fails():
     """Ensure OpenCV cascade fallback works when RetinaFace raises exception."""
     tile = np.zeros((100, 100, 3), dtype=np.uint8)
@@ -121,25 +128,32 @@ def test_detect_faces_falls_back_to_opencv_when_retinaface_fails():
     
     from detection_core import _detect_faces_opencv
 
     # Provide a simple cascade that doesn't error for fallback
     cascade = Mock()
     cascade.empty.return_value = False
     cascade.detectMultiScale.return_value = []
 
     with patch("detection_core._get_opencv_cascade", return_value=cascade):
         # Inject a fake deepface module that raises to trigger fallback
-        fake_deepface = Mock()
-        fake_deepface.DeepFace = Mock()
-        fake_deepface.DeepFace.extract_faces = Mock(side_effect=RuntimeError("TensorFlow error"))
+        import types
         import sys
-        with patch.dict(sys.modules, {"deepface": fake_deepface}):
+
+        module = types.ModuleType("deepface")
+
+        class DeepFace:
+            @staticmethod
+            def extract_faces(*_args, **_kwargs):
+                raise RuntimeError("TensorFlow error")
+
+        module.DeepFace = DeepFace
+        with patch.dict(sys.modules, {"deepface": module}):
             detections = _detect_faces_retinaface(tile)
     
     # Fallback should return OpenCV results (might be empty on blank test image)
     # Just verify it doesn't crash and returns a list
     assert isinstance(detections, list)
 
 
 def test_detect_faces_opencv_returns_detection_contract():
     frame = np.zeros((80, 100, 3), dtype=np.uint8)
     cascade = Mock()
```
