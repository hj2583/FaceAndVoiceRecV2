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

Split each pipeline into a reader stage and a compute stage, each running on its own background thread and connected by bounded queues; the writer/display stage runs inline on the thread that calls `run_threaded_pipeline` (the caller's own thread), not a third worker thread:

```mermaid
flowchart LR
    A[Reader Thread\ncv2.VideoCapture.read] -- Queue A --> B[Compute Thread\ndetect/track/recognize]
    B -- Queue B --> C[Caller's Thread\nVideoWriter.write or imshow, inline]
```

- **Reader thread**: only calls `cap.read()`, pushes raw frames into Queue A.
- **Compute thread** (single instance, not a pool): pulls a frame, runs the existing detection/tracking/recognition logic unchanged, pushes the annotated result into Queue B. Kept single-threaded because the face tracker is stateful and must observe frames strictly in order — parallelizing this stage would corrupt track IDs.
- **Writer/display stage**: runs inline on the caller's own thread. It pulls annotated frames from Queue B and writes them to the output video (offline) or renders them (realtime), in order. Running it on the caller's thread (rather than a third worker thread) is required so its side effects — Streamlit calls in the offline Streamlit UI path, `cv2.imshow`/`cv2.waitKey` in the realtime GUI path — execute with the caller's real thread context (e.g. Streamlit's `ScriptRunContext`, which only exists on the thread Streamlit started the script on).

OpenCV's decode/encode calls and ONNX Runtime's `session.run()` release the GIL during their native/CUDA work, so the reader and compute threads run genuinely concurrently with the caller's writer loop on multi-core CPU + GPU: decoding frame N+1 happens while frame N is on the GPU, while frame N-1 is still being written/displayed on the caller's thread.

Order is preserved automatically: each queue has exactly one producer and one consumer, so FIFO order equals frame order. No reordering logic is needed anywhere.

## `video_processor.py` (offline) specifics

- Queue A (raw frames) and Queue B (annotated frames) are both **bounded and blocking** (`maxsize=8`). No frames are ever dropped — output video length/frame count must exactly match the source. If the compute or writer stage falls behind, the reader blocks on `queue.put()` (natural backpressure), bounding memory to ~8 buffered frames instead of the whole video.
- `TILED_DETECTION_INTERVAL` / `RECOGNITION_INTERVAL` skip logic is unchanged and runs inside the compute thread exactly as today — only the *scheduling/threading* changes, not the detection/recognition cadence.
- End-of-video: the reader pushes an internal sentinel object (not `None` — `read_frame` returning `None` is the external end-of-stream signal, but the value forwarded between queues is a dedicated sentinel object so it can never collide with a real frame value) into Queue A once `cap.read()` returns `False`. The compute thread forwards the sentinel to Queue B after flushing its current work. The caller's inline writer loop sees the sentinel, finalizes the `VideoWriter` and log file, and the pipeline function returns.
- Any exception in the reader/compute threads (including `BaseException` subclasses such as `SystemExit`/framework-specific control-flow exceptions, not just `Exception`) sets a shared `threading.Event` (`stop_event`), logs the traceback, and stores the exception in a shared slot. If an interruption (e.g. `KeyboardInterrupt`) arrives on the caller's thread while it is waiting for/writing items, the caller likewise sets `stop_event` and records the exception. All stages check `stop_event` on every queue-wait iteration and unwind cleanly instead of hanging; the caller joins the worker threads via a short, interruptible poll loop rather than a single blocking `join()`. The caller then re-raises the stored exception if one occurred, so callers observe failures exactly as before (no silent hangs, no swallowed errors). A dedicated `PipelineStop` exception lets any stage request a clean, silent shutdown instead (no error log, no re-raised exception) — used by the realtime quit path (window close / q / Esc).

## `realtime.py` (live camera) specifics

- Same reader/compute background-thread shape (with the writer running inline on the caller's thread), but Queue A uses `maxsize=1` and **drops the oldest frame when full** (overwrite instead of block) — a live camera must never make capture stall waiting for a slow consumer; showing the newest frame is more important than showing every frame. This is a small behavior change (occasional frame skip under load) but strictly better than today, where a slow frame currently stalls the capture call itself and can starve the camera driver's internal buffer.
- Queue B (annotated frames for display) also uses `maxsize=1` with overwrite, so display always shows the latest processed frame instead of a backlog of stale ones.
- The existing audio VAD background thread is untouched; it already runs independently of the video loop.
- Shutdown (window close / q / Esc) raises `PipelineStop` from the inline writer stage, which sets `stop_event` and makes `run_threaded_pipeline` return normally (no exception, no error log). Ctrl+C (`KeyboardInterrupt`) arriving on the caller's thread is treated like any other caller-side exception: `stop_event` is set and re-raised after joining. Either way, the reader/compute threads exit their loops and are joined (via a short, interruptible poll) before the function returns — no daemon-thread leaks.

## Error Handling

- A shared `stop_event` (`threading.Event`) plus a shared `first_exception` slot (protected by a lock) coordinate shutdown across the reader thread, the compute thread, and the caller's inline writer loop.
- Any stage that raises a `BaseException` (not just `Exception`): logs the traceback, stores the exception in `first_exception` (only if not already set), sets `stop_event`, and stops its loop. A stage raising `PipelineStop` instead skips the log and the `first_exception` slot entirely, signaling a clean, silent shutdown.
- Queue operations use short timeouts (e.g. `queue.get(timeout=0.5)` / `queue.put(timeout=0.5)`) inside a loop that also checks `stop_event`, so no thread blocks forever if another stage has already died.
- The caller joins the reader/compute threads on exit (normal, `PipelineStop`, or error) using a short, interruptible poll loop (`while thread.is_alive(): thread.join(0.1)`) rather than a single blocking `join()`, so an interruption (e.g. `KeyboardInterrupt`) arriving while the caller waits does not itself get stuck; it then re-raises `first_exception` if set, preserving today's caller-facing error behavior.

## Testing

- New unit tests for the reader/compute thread wiring plus the caller's inline writer loop, using a mocked `cv2.VideoCapture` feeding synthetic frames (following existing fixture patterns in `tests/test_video_processor.py` / `tests/test_realtime_runtime.py`):
  - Frame order is preserved end-to-end.
  - Frame count in equals frame count out for the offline path (no drops).
  - Sentinel/end-of-stream and normal shutdown are handled without hangs.
  - An injected exception in one stage propagates to the caller and stops all threads (no deadlock).
  - Realtime drop-oldest queue behavior under a slow consumer (verify newest frame wins, no stall).
- Manual verification: run `video_processor.py` on a sample clip before/after this change, compare wall-clock processing time, and observe `nvidia-smi` GPU utilization during the run.
