from types import SimpleNamespace

from runtime_status import get_runtime_status


def _fake_torch(available, name="Test GPU"):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_device_name=lambda _index: name,
        )
    )


def test_reports_torch_gpu_and_face_cpu_as_mixed_runtime():
    status = get_runtime_status(
        torch_module=_fake_torch(True, "RTX Test"),
        face_provider_getter=lambda: ["CPUExecutionProvider"],
    )
    assert status == {
        "torch": "CUDA (RTX Test)",
        "face": "CPU (CUDA provider unavailable)",
    }


def test_reports_both_cpu():
    status = get_runtime_status(
        torch_module=_fake_torch(False),
        face_provider_getter=lambda: ["CPUExecutionProvider"],
    )
    assert status == {"torch": "CPU", "face": "CPU (CUDA provider unavailable)"}


def test_reports_cuda_face_provider():
    status = get_runtime_status(
        torch_module=_fake_torch(True),
        face_provider_getter=lambda: ["CUDAExecutionProvider"],
    )
    assert status["face"] == "CUDA provider available (active session unverified)"


def test_reports_torch_unavailable_when_cuda_probe_raises_but_face_probe_still_runs():
    class FailingCuda:
        def is_available(self):
            raise RuntimeError("cuda probe failed")

        def get_device_name(self, _index):
            return "ignored"

    status = get_runtime_status(
        torch_module=SimpleNamespace(cuda=FailingCuda()),
        face_provider_getter=lambda: ["CPUExecutionProvider"],
    )

    assert status["torch"] == "Unavailable"
    assert status["face"] == "CPU (CUDA provider unavailable)"


def test_reports_face_runtime_unavailable_when_provider_detection_fails():
    def fail():
        raise RuntimeError("provider load failed")

    status = get_runtime_status(torch_module=_fake_torch(False), face_provider_getter=fail)
    assert status["face"] == "Unavailable"
