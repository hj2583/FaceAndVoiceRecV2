import numpy as np
import sys
import importlib
import inspect
from typing import Callable, Sequence


def _install_numpy_compat_aliases() -> None:
    # NumPy 1.26 removes legacy aliases some dependencies still expect.
    if not hasattr(np, "long"):
        np.long = np.int_
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "complex"):
        np.complex = complex


_install_numpy_compat_aliases()

import streamlit as st

from app.ui.status import get_runtime_status
from face_voice_rec.config import initialize_runtime_paths
from face_voice_rec.storage.database import init_db


TAB_PAGES: tuple[tuple[str, str], ...] = (
    ("📹 Realtime", "app.pages.realtime"),
    ("🎬 Video", "app.pages.video"),
    ("👤 Face Database", "app.pages.face_database"),
    ("🔊 Audio Logs", "app.pages.audio_logs"),
    ("✅ Attendance", "app.pages.attendance"),
    ("📝 Transcripts", "app.pages.transcripts"),
    ("🎙️ Voice Enrollment", "app.pages.voice_enrollment"),
)


class _LazyPageModule:
    def __init__(self, module_path: str) -> None:
        self._module_path = module_path

    def render(self) -> None:
        importlib.import_module(self._module_path).render()


realtime = _LazyPageModule("app.pages.realtime")
video = _LazyPageModule("app.pages.video")
face_database = _LazyPageModule("app.pages.face_database")
audio_logs = _LazyPageModule("app.pages.audio_logs")
attendance = _LazyPageModule("app.pages.attendance")
transcripts = _LazyPageModule("app.pages.transcripts")
voice_enrollment = _LazyPageModule("app.pages.voice_enrollment")


_MODULE_NAME_OVERRIDES = {
    "app.pages.realtime": realtime,
    "app.pages.video": video,
    "app.pages.face_database": face_database,
    "app.pages.audio_logs": audio_logs,
    "app.pages.attendance": attendance,
    "app.pages.transcripts": transcripts,
    "app.pages.voice_enrollment": voice_enrollment,
}


def supports_stateful_tabs(tabs_callable: Callable[..., object]) -> bool:
    try:
        parameters = inspect.signature(tabs_callable).parameters
    except (TypeError, ValueError):
        return False
    return "on_change" in parameters and "key" in parameters


def _render_tab_pages(
    tab_containers: Sequence[object],
    module_paths: Sequence[str],
    *,
    active_only: bool,
    import_module: Callable[[str], object] = importlib.import_module,
) -> None:
    for tab_container, module_path in zip(tab_containers, module_paths):
        with tab_container:
            if active_only and not getattr(tab_container, "open", False):
                continue
            page_module = import_module(module_path)
            page_module.render()


def _import_page_module(module_path: str) -> object:
    # Keep old tests and fallback behavior patchable without eager imports.
    return _MODULE_NAME_OVERRIDES.get(module_path, importlib.import_module(module_path))


@st.cache_resource
def initialize_application_database() -> None:
    init_db()


def render_application() -> None:
    st.title("🎥 AI Face + Active Speaker Recognition")

    runtime_status = get_runtime_status()
    st.caption(f"Audio models (PyTorch): {runtime_status['torch']}")
    st.caption(f"Face models (ONNX Runtime): {runtime_status['face']}")

    labels = [label for label, _module_path in TAB_PAGES]
    module_paths = [module_path for _label, module_path in TAB_PAGES]

    if supports_stateful_tabs(st.tabs):
        tabs = st.tabs(labels, on_change="rerun", key="main-tabs")
        _render_tab_pages(
            tabs,
            module_paths,
            active_only=True,
            import_module=_import_page_module,
        )
    else:
        tabs = st.tabs(labels)
        _render_tab_pages(
            tabs,
            module_paths,
            active_only=False,
            import_module=_import_page_module,
        )


def main() -> None:
    st.set_page_config(
        page_title="AI Face + Audio Recognition",
        page_icon="🎥",
        layout="wide",
    )

    initialize_runtime_paths()
    initialize_application_database()

    navigation = st.navigation(
        [
            st.Page(
                render_application,
                title="AI Face + Active Speaker Recognition",
                default=True,
            )
        ],
        position="hidden",
    )
    navigation.run()


def _is_running_via_streamlit() -> bool:
    argv = " ".join(sys.argv).lower()
    if "streamlit" in argv and " run " in f" {argv} ":
        return True

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


if __name__ == "__main__" or _is_running_via_streamlit():
    main()
