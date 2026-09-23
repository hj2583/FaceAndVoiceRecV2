# GPU Utilization: Decoupled Producer/Consumer Pipeline

## Problem

Both the offline video processing pipeline (`video_processor.py`) and the live camera pipeline (`realtime.py`) run a single synchronous loop per frame:

```
decode (CPU) -> detect (GPU) -> track (CPU) -> recognize (GPU) -> encode/display (CPU)
```

Every stage blocks the next within the same thread. As a result:

- The GPU sits idle while the CPU decodes/encodes frames.
- The CPU sits idle while waiting on GPU inference calls (ONNX Runtime CUDA).
- Reported symptoms: GPU utilization stuck around 10-20%, video processing wall-clock time is very long, and live camera FPS occasionally drops.

The GPU is not the bottleneck — the lack of overlap between CPU and GPU work is.

## Goals

- Increase GPU utilization during offline video processing and reduce total wall-clock processing time.
- Reduce/avoid FPS stalls in the live camera pipeline caused by one slow stage blocking capture.
- Preserve exact current detection/recognition/tracking behavior and output correctness (frame order, frame count, log/output parity) — this is a scheduling/concurrency change only, not an algorithm change.

## Non-Goals

- Batching multiple frames or face crops into single GPU inference calls (out of scope for this change; may be a future follow-up).
- Multiprocessing / multiple GPU compute workers.
- Hardware-accelerated video decode/encode (NVDEC/NVENC).

## Architecture

Split each pipeline into 3 stages connected by bounded queues, each running in its own thread:

```mermaid
flowchart LR
    A[Reader Thread\ncv2.VideoCapture.read] -- Queue A --> B[Compute Thread\ndetect/track/recognize]
    B -- Queue B --> C[Writer/Display Thread\nVideoWriter.write or imshow]
```

- **Reader thread**: only calls `cap.read()`, pushes raw frames into Queue A.
- **Compute thread** (single instance, not a pool): pulls a frame, runs the existing detection/tracking/recognition logic unchanged, pushes the annotated result into Queue B. Kept single-threaded because the face tracker is stateful and must observe frames strictly in order — parallelizing this stage would corrupt track IDs.
- **Writer/display thread**: pulls annotated frames and writes them to the output video (offline) or renders them (realtime), in order.

OpenCV's decode/encode calls and ONNX Runtime's `session.run()` release the GIL during their native/CUDA work, so these three threads run genuinely concurrently on multi-core CPU + GPU: decoding frame N+1 happens while frame N is on the GPU, while frame N-1 is still being encoded/written.

Order is preserved automatically: each queue has exactly one producer and one consumer, so FIFO order equals frame order. No reordering logic is needed anywhere.

## `video_processor.py` (offline) specifics

- Queue A (raw frames) and Queue B (annotated frames) are both **bounded and blocking** (`maxsize=8`). No frames are ever dropped — output video length/frame count must exactly match the source. If the compute or writer stage falls behind, the reader blocks on `queue.put()` (natural backpressure), bounding memory to ~8 buffered frames instead of the whole video.
- `TILED_DETECTION_INTERVAL` / `RECOGNITION_INTERVAL` skip logic is unchanged and runs inside the compute thread exactly as today — only the *scheduling/threading* changes, not the detection/recognition cadence.
- End-of-video: the reader pushes a sentinel (`None`) into Queue A once `cap.read()` returns `False`. The compute thread forwards the sentinel to Queue B after flushing its current work. The writer thread sees the sentinel, finalizes the `VideoWriter` and log file, and the pipeline function returns.
- Any exception in any thread sets a shared `threading.Event` (`stop_event`), logs the traceback, and stores the exception in a shared slot. The other threads check `stop_event` on every queue-wait iteration and unwind cleanly instead of hanging. The main function joins all threads and re-raises the stored exception if one occurred, so callers observe failures exactly as before (no silent hangs, no swallowed errors).

## `realtime.py` (live camera) specifics

- Same 3-thread shape, but Queue A uses `maxsize=1` and **drops the oldest frame when full** (overwrite instead of block) — a live camera must never make capture stall waiting for a slow consumer; showing the newest frame is more important than showing every frame. This is a small behavior change (occasional frame skip under load) but strictly better than today, where a slow frame currently stalls the capture call itself and can starve the camera driver's internal buffer.
- Queue B (annotated frames for display) also uses `maxsize=1` with overwrite, so display always shows the latest processed frame instead of a backlog of stale ones.
- The existing audio VAD background thread is untouched; it already runs independently of the video loop.
- Shutdown (window close / Ctrl+C) sets the same `stop_event`; reader/compute/writer threads exit their loops and are joined before the function returns — no daemon-thread leaks.

## Error Handling

- A shared `stop_event` (`threading.Event`) plus a shared `first_exception` slot (protected by a lock) coordinate shutdown across all three threads.
- Any thread that raises: logs the traceback, stores the exception in `first_exception` (only if not already set), sets `stop_event`, and stops its loop.
- Queue operations use short timeouts (e.g. `queue.get(timeout=0.5)` / `queue.put(timeout=0.5)`) inside a loop that also checks `stop_event`, so no thread blocks forever if another stage has already died.
- The orchestrating function joins all threads on exit (normal or error) and re-raises `first_exception` if set, preserving today's caller-facing error behavior.

## Testing

- New unit tests for the reader/compute/writer thread wiring, using a mocked `cv2.VideoCapture` feeding synthetic frames (following existing fixture patterns in `tests/test_video_processor.py` / `tests/test_realtime_runtime.py`):
  - Frame order is preserved end-to-end.
  - Frame count in equals frame count out for the offline path (no drops).
  - Sentinel/end-of-stream and normal shutdown are handled without hangs.
  - An injected exception in one stage propagates to the caller and stops all threads (no deadlock).
  - Realtime drop-oldest queue behavior under a slow consumer (verify newest frame wins, no stall).
- Manual verification: run `video_processor.py` on a sample clip before/after this change, compare wall-clock processing time, and observe `nvidia-smi` GPU utilization during the run.
