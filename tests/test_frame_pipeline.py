import queue
import threading
import time

import pytest

from frame_pipeline import run_threaded_pipeline


def test_preserves_order_and_count_with_blocking_queues():
    items = list(range(20))
    written = []

    def read_frame(_it=iter(items)):
        return next(_it, None)

    def process_frame(item):
        return item * 2

    def write_frame(item):
        written.append(item)

    run_threaded_pipeline(read_frame, process_frame, write_frame, queue_maxsize=3)

    assert written == [item * 2 for item in items]


def test_propagates_exception_from_process_stage_and_stops_all_threads():
    def read_frame(_it=iter(range(5))):
        return next(_it, None)

    def process_frame(item):
        if item == 2:
            raise ValueError("boom")
        return item

    written = []

    with pytest.raises(ValueError, match="boom"):
        run_threaded_pipeline(read_frame, process_frame, written.append, queue_maxsize=2)

    # No thread should be left running.
    time.sleep(0.1)
    active_names = {t.name for t in threading.enumerate()}
    assert not any(name.startswith("frame-pipeline-") for name in active_names)


def test_drop_oldest_keeps_latest_item_when_consumer_is_slow():
    produced = list(range(50))
    write_started = threading.Event()
    release_write = threading.Event()
    written = []

    def read_frame(_it=iter(produced)):
        return next(_it, None)

    def process_frame(item):
        return item

    def write_frame(item):
        if item == produced[0]:
            write_started.set()
            assert release_write.wait(timeout=2.0)
        written.append(item)

    def _run():
        run_threaded_pipeline(
            read_frame,
            process_frame,
            write_frame,
            drop_oldest=True,
            poll_interval=0.05,
        )

    thread = threading.Thread(target=_run)
    thread.start()
    assert write_started.wait(timeout=2.0)
    release_write.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    # With drop-oldest queues the consumer cannot see every produced item,
    # but it must see the first item and the very last one.
    assert written[0] == produced[0]
    assert written[-1] == produced[-1]
    assert len(written) < len(produced)
