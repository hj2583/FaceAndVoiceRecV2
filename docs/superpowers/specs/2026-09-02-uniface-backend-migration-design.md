# UniFace backend migration design

## Summary

This design adds a pluggable face-backend layer so the project can evaluate and adopt UniFace without breaking the current recognition pipeline. The main objective is to reduce dependency risk from the DeepFace/TensorFlow path while preserving the existing NumPy indexing, SQLite persistence, tracking, and recognition workflow.

The migration is intentionally scoped to the face detection and recognition backend layer. The custom similarity index in the project stays intact, because its logic is already independent from the model provider and is not the root cause of the reliability issue.

## Current situation

The project currently binds face detection and embedding extraction directly to DeepFace:

- detection calls DeepFace.extract_faces(..., detector_backend="retinaface") in detection_core.py
- recognition calls DeepFace.build_model("ArcFace") and DeepFace.represent(...) in face_core.py
- thresholds and matching assumptions are defined in config.py
- the repository already contains fallback logic for RetinaFace failures and OpenCV detection fallback, which confirms the dependency is brittle in some environments

This matches the identified project constraint: the main problem is the face inference backend, not the database or similarity index design.

## Goals

1. Add an adapter layer so the app can use UniFace, DeepFace, or a lightweight fallback backend.
2. Keep the current matching/index model stable while testing a replacement backend.
3. Preserve recognition workflow compatibility with minimal code disruption.
4. Reduce TensorFlow/DeepFace dependency pressure on CPU-only or constrained environments.

## Non-goals

1. Rewriting the entire tracking or speaker pipeline.
2. Replacing the SQLite/NumPy indexing model in the same change.
3. Re-enrolling every person before backend compatibility is validated.
4. Broad cleanup or large refactors unrelated to the face backend.

## Design principles

- Keep the public behavior of the system stable.
- Prefer adapter boundaries over direct backend coupling.
- Validate compatibility before changing the default backend.
- Preserve fallback behavior during migration.

## Proposed architecture

### Backend interface

Add a small abstraction with the following operations:

- detect_faces(image) -> list of face bbox objects
- extract_embedding(face_crop) -> normalized embedding vector
- is_available() -> bool
- backend_name -> string

This abstraction sits between the app logic and the actual model runtime.

### Implementation package

Create a new file, such as face_backend.py, with the following components:

- DeepFaceBackend
- UniFaceBackend
- OpenCVFallbackBackend
- FaceBackendFactory

The factory chooses the preferred backend by environment and availability.

### Detection flow

1. The app requests face boxes from the backend adapter.
2. The adapter tries the preferred backend.
3. If the preferred backend raises an exception or returns no valid faces, it falls back to the next backend.
4. The detection pipeline continues with the returned box list.

### Recognition flow

1. A tracked face crop is passed to the backend adapter.
2. The adapter calls extract_embedding(face_crop).
3. The result must conform to the current embedding contract:
   - vector length matches EMBEDDING_DIM
   - value type is float32
   - vector is normalized before storage and comparison
4. The result continues through the current NumPy search logic in face_core.py.

## Compatibility requirements

Before switching UniFace to the default backend, these checks must pass:

- embedding dimension matches the configured value in config.py
- vector normalization behavior is consistent with the current _normalize helper
- threshold calibration remains stable for the current known-face dataset
- ambiguous-match behavior still makes sense with the existing AMBIGUITY_MARGIN

If UniFace produces a different embedding scale or a different model family, the app should not silently use it. The migration must either:

- re-enroll known faces and retune thresholds, or
- keep DeepFace as the default until the compatibility gap is resolved

## Suggested migration sequence

### Phase 1: adapter layer and detection fallback

- Add backend abstraction.
- Implement DeepFace detector as the initial adapter.
- Implement UniFace detector behind the same interface.
- Keep OpenCV fallback as last resort.
- Validate with the existing detection_core tests.

### Phase 2: recognition backend validation

- Add UniFace embedding extraction with a guarded fallback.
- Log backend name and embedding size when generating embeddings.
- Validate a sample of known faces and unknown faces.
- Compare score distributions against the current threshold values.

### Phase 3: default switch

- Select UniFace as the default only after compatibility checks pass.
- Keep DeepFace fallback enabled for failover.
- Add explicit environment variables or config settings for backend selection.

## Configuration plan

Add backend-config settings in config.py, for example:

- FACE_BACKEND = "uniface"
- FACE_BACKEND_FALLBACK = "deepface"
- FACE_BACKEND_STRICT_COMPATIBILITY = True

This keeps the migration reversible and easy to diagnose in production.

## Testing plan

The tests should verify behavior at the backend boundary rather than the entire system end-to-end.

### Detection tests

- detection_core continues to pass its current geometric tests
- backend adapter returns boxes in the same coordinate contract
- fallback path still triggers when the preferred backend fails

### Recognition tests

- embedding shape matches EMBEDDING_DIM
- embedding extraction returns None for invalid inputs
- normalized output remains finite and stable
- compatibility check fails loudly when vector dimensions do not match expectations

### Manual validation

- run the app with a known face in the current dataset
- compare recognition score distribution before and after switch
- verify unknown face handling remains stable

## Risks and mitigations

### Risk: embedding mismatch

Mitigation: enforce strict validation and refuse to use incompatible embeddings.

### Risk: runtime instability from missing dependencies

Mitigation: lazy import, runtime availability checks, and fallback ordering.

### Risk: threshold miscalibration

Mitigation: compare known-face score distributions before defaulting the backend.

### Risk: silent quality regression

Mitigation: backend logging and explicit compatibility checks in validation mode.

## Recommended decision

Proceed with a dual-backend adapter and keep the current index pipeline unchanged until UniFace compatibility is validated. This minimizes risk and creates a safe migration path without destabilizing the app.

## Follow-up

After review, the next step is to create a detailed implementation plan for the adapter, fallback logic, and validation steps.
