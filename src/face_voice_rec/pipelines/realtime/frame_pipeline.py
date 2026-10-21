"""Threaded reader/compute pipeline with an inline writer stage.

run_threaded_pipeline runs a reader and a compute stage on two background
threads, connected by bounded queues; the writer stage runs inline on the
caller's own thread so its side effects keep the caller's thread context.
"""

import logging
import queue
import threading

logger = logging.getLogger(__name__)

_SENTINEL = object()

# Short timeout used when joining worker threads from the caller so that the
# caller's own wait loop stays interruptible (see run_threaded_pipeline).
_JOIN_POLL_INTERVAL = 0.1


class PipelineStop(Exception):
    """Raised by any stage to request a clean, silent pipeline shutdown.

    Unlike other exceptions, a PipelineStop is not logged as an error and is
    not re-raised by run_threaded_pipeline: it simply stops all stages and
    causes run_threaded_pipeline to return normally.
    """


class PipelineErrorBox:
    """Shared single-slot holder for the first exception raised by any stage."""

    def __init__(self):
        self._lock = threading.Lock()
        self._exception = None

    def set(self, exc):
        with self._lock:
            if self._exception is None:
                self._exception = exc

    def get(self):
        with self._lock:
            return self._exception


def _run_stage(name, stop_event, error_box, func):
    try:
        func()
    except PipelineStop:
        stop_event.set()
    except BaseException as exc:  # noqa: BLE001 - recorded and re-raised by the caller
        logger.exception("Pipeline stage '%s' failed", name)
        error_box.set(exc)
        stop_event.set()


def _put_latest(q, item):
    """Put item on a maxsize=1 queue, dropping any previously queued item."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def _enqueue(q, item, stop_event, drop_oldest, poll_interval, first_consumed=None):
    # The end-of-stream sentinel must never be dropped in favor of a data
    # item still waiting to be consumed, even in drop_oldest mode, or the
    # last real item would be silently lost. Only real items use drop-oldest
    # overwrite semantics; the sentinel always waits for a free slot.
    if drop_oldest and item is not _SENTINEL:
        if first_consumed is None or first_consumed.is_set():
            _put_latest(q, item)
            return True
        # Deterministic startup handshake: until the consumer has actually
        # dequeued at least one item from this queue, block on a full queue
        # instead of overwriting it, so the very first item can never be
        # silently dropped by a producer that outruns the consumer's startup.
        while not stop_event.is_set():
            if first_consumed.is_set():
                _put_latest(q, item)
                return True
            try:
                q.put(item, timeout=poll_interval)
                return True
            except queue.Full:
                continue
        return False
    while not stop_event.is_set():
        try:
            q.put(item, timeout=poll_interval)
            return True
        except queue.Full:
            continue
    return False


def _dequeue(q, stop_event, poll_interval, first_consumed=None):
    """Returns (True, item) on success, (False, None) if stop_event fired first."""
    while not stop_event.is_set():
        try:
            item = q.get(timeout=poll_interval)
            if first_consumed is not None:
                first_consumed.set()
            return True, item
        except queue.Empty:
            continue
    return False, None


def run_threaded_pipeline(
    read_frame,
    process_frame,
    write_frame,
    queue_maxsize=8,
    drop_oldest=False,
    poll_interval=0.5,
):
    """Run read_frame -> process_frame -> write_frame across two worker threads
    plus the caller's own thread.

    read_frame() returns the next item, or None to signal end-of-stream.
    process_frame(item) transforms one item at a time, in strict input order.
    write_frame(item) consumes one processed item at a time, in order.

    The reader and compute stages each run on a dedicated background thread
    (named "frame-pipeline-reader" / "frame-pipeline-compute"), giving real
    CPU/GPU overlap: decoding frame N+1 can happen while frame N is being
    processed. The writer stage runs INLINE on the thread that calls
    run_threaded_pipeline (no third worker thread) so that write_frame's side
    effects (Streamlit calls, GUI calls, VideoWriter.write, etc.) execute with
    the caller's own thread context.

    A stage (including the inline writer) may raise PipelineStop to request a
    clean, silent shutdown: all stages stop, all worker threads are joined,
    and run_threaded_pipeline returns normally (no exception, no error log).

    When drop_oldest is False (default), both internal queues are bounded and
    blocking with maxsize=queue_maxsize: no item is ever dropped, at the cost
    of the reader blocking if the compute/writer stages fall behind.

    When drop_oldest is True, both internal queues have maxsize=1 and silently
    replace the queued item when full: the newest item always wins. As a
    one-time startup handshake, the reader/compute stage blocks on a full
    queue until the downstream stage has dequeued at least one item from it,
    so the very first item can never be silently dropped by a producer that
    outruns the consumer's startup; after that first item has been consumed,
    no stage ever blocks waiting for a slow downstream consumer again.

    If the calling thread is interrupted (e.g. KeyboardInterrupt) while
    waiting for items to write, or write_frame itself raises, stop_event is
    set and all worker threads are joined (with a short, interruptible poll)
    before the exception is re-raised.

    Raises the first exception encountered in any stage (including the
    caller's own writer loop), after all worker threads have been joined.
    """
    maxsize = 1 if drop_oldest else queue_maxsize
    raw_q = queue.Queue(maxsize=maxsize)
    out_q = queue.Queue(maxsize=maxsize)

    stop_event = threading.Event()
    error_box = PipelineErrorBox()
    raw_first_consumed = threading.Event() if drop_oldest else None
    out_first_consumed = threading.Event() if drop_oldest else None

    def _reader():
        while not stop_event.is_set():
            frame = read_frame()
            if frame is None:
                _enqueue(raw_q, _SENTINEL, stop_event, drop_oldest, poll_interval)
                return
            if not _enqueue(
                raw_q, frame, stop_event, drop_oldest, poll_interval, raw_first_consumed
            ):
                return

    def _compute():
        while True:
            ok, frame = _dequeue(raw_q, stop_event, poll_interval, raw_first_consumed)
            if not ok:
                return
            if frame is _SENTINEL:
                _enqueue(out_q, _SENTINEL, stop_event, drop_oldest, poll_interval)
                return
            processed = process_frame(frame)
            if not _enqueue(
                out_q, processed, stop_event, drop_oldest, poll_interval, out_first_consumed
            ):
                return

    threads = [
        threading.Thread(
            target=_run_stage,
            args=(name, stop_event, error_box, stage),
            name=f"frame-pipeline-{name}",
            daemon=True,
        )
        for name, stage in (
            ("reader", _reader),
            ("compute", _compute),
        )
    ]

    for thread in threads:
        thread.start()

    try:
        while True:
            ok, item = _dequeue(out_q, stop_event, poll_interval, out_first_consumed)
            if not ok:
                break
            if item is _SENTINEL:
                break
            try:
                write_frame(item)
            except PipelineStop:
                stop_event.set()
                break
    except BaseException as exc:  # noqa: BLE001 - recorded and re-raised below
        stop_event.set()
        error_box.set(exc)
    finally:
        for thread in threads:
            while thread.is_alive():
                thread.join(_JOIN_POLL_INTERVAL)

    exc = error_box.get()
    if exc is not None:
        raise exc

