"""Load .npy embeddings without allowing arbitrary pickle code execution."""

import pickle
from pathlib import Path

import numpy as np

_ALLOWED_GLOBALS = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Forbidden pickle global: {module}.{name}")


def load_embedding_array(path) -> np.ndarray:
    """Load a .npy file; legacy object arrays are unpickled with a numpy-only allowlist."""
    try:
        return np.load(Path(path), allow_pickle=False)
    except ValueError as error:
        if "allow_pickle" not in str(error):
            raise
    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError("Unsupported .npy format version")
        return _RestrictedUnpickler(handle).load()
