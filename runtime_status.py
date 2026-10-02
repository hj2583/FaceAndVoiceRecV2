def get_runtime_status(torch_module=None, face_provider_getter=None):
    status = {
        "torch": "Unavailable",
        "face": "Unavailable",
    }

    # Detect PyTorch runtime availability independently from ONNX Runtime.
    try:
        if torch_module is None:
            import torch as torch_module

        if torch_module.cuda.is_available():
            status["torch"] = f"CUDA ({torch_module.cuda.get_device_name(0)})"
        else:
            status["torch"] = "CPU"
    except Exception:
        status["torch"] = "Unavailable"

    # Detect ONNX Runtime provider availability independently from PyTorch.
    try:
        if face_provider_getter is None:
            from face_backend import get_cuda_providers

            face_provider_getter = get_cuda_providers

        providers = face_provider_getter()
        if providers and "CUDAExecutionProvider" in providers:
            status["face"] = "CUDA provider available (active session unverified)"
        elif providers and "CPUExecutionProvider" in providers:
            status["face"] = "CPU (CUDA provider unavailable)"
        else:
            status["face"] = "Unavailable"
    except Exception:
        status["face"] = "Unavailable"

    return status
