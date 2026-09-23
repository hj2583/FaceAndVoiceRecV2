import logging
import queue
import threading

logger = logging.getLogger(__name__)

_SENTINEL = object()


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
    except Exception as exc:  # noqa: BLE001 - recorded and re-raised by the caller
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
    """Run read_frame -> process_frame -> write_frame across three threads.

    read_frame() returns the next item, or None to signal end-of-stream.
    process_frame(item) transforms one item at a time, in strict input order.
    write_frame(item) consumes one processed item at a time, in order.

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

    Raises the first exception encountered in any stage, after all three
    threads have stopped and been joined.
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

    def _writer():
        while True:
            ok, frame = _dequeue(out_q, stop_event, poll_interval, out_first_consumed)
            if not ok:
                return
            if frame is _SENTINEL:
                return
            write_frame(frame)

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
            ("writer", _writer),
        )
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    exc = error_box.get()
    if exc is not None:
        raise exc
