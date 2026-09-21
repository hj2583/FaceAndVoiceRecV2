import numpy as np

# NumPy 2.x removed the legacy scalar aliases such as np.long.
# Some older dependencies still reference them at import time.
# Reintroduce the compatibility aliases so the app keeps working
# with third-party packages that have not yet been updated.
if not hasattr(np, "long"):
    np.long = np.int_

if not hasattr(np, "float"):
    np.float = float

if not hasattr(np, "complex"):
    np.complex = complex
