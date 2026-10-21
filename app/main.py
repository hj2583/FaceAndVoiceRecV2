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


def get_runtime_status() -> dict[str, str]:
    status_module = importlib.import_module("app.ui.status")
    return status_module.get_runtime_status()


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
    loaded_module_paths: set[str] | None = None,
    import_module: Callable[[str], object] = importlib.import_module,
) -> None:
    for tab_container, module_path in zip(tab_containers, module_paths):
        with tab_container:
            if active_only:
                if loaded_module_paths is None:
                    should_render = getattr(tab_container, "open", False)
                else:
                    should_render = module_path in loaded_module_paths
                if not should_render:
                    continue
            page_module = import_module(module_path)
            page_module.render()


@st.cache_resource
def initialize_application_database() -> None:
    init_db()


def render_application() -> None:
    st.title("🎥 AI Face + Active Speaker Recognition")

    session_state = getattr(st, "session_state", None)
    if session_state is None:
        session_state = {}

    runtime_status = session_state.get("runtime_status")
    runtime_status_error = session_state.get("runtime_status_error")

    check_runtime_button = getattr(st, "button", None)
    should_probe_runtime = False
    if runtime_status is None:
        if callable(check_runtime_button):
            should_probe_runtime = check_runtime_button("Check runtime")

    if runtime_status is None and should_probe_runtime:
        try:
            runtime_status = get_runtime_status()
            session_state["runtime_status"] = runtime_status
            session_state.pop("runtime_status_error", None)
            runtime_status_error = None
        except Exception as exc:
            runtime_status = None
            runtime_status_error = str(exc)
            session_state["runtime_status_error"] = runtime_status_error

    if runtime_status is not None:
        st.caption(f"Audio models (PyTorch): {runtime_status['torch']}")
        st.caption(f"Face models (ONNX Runtime): {runtime_status['face']}")
    elif runtime_status_error:
        st.caption("Audio models (PyTorch): unavailable")
        st.caption("Face models (ONNX Runtime): unavailable")
        st.caption(f"Runtime status error: {runtime_status_error}")

    labels = [label for label, _module_path in TAB_PAGES]
    module_paths = [module_path for _label, module_path in TAB_PAGES]
    import_module = importlib.import_module

    if supports_stateful_tabs(st.tabs):
        tabs = st.tabs(labels, on_change="rerun", key="main-tabs")
        loaded_tab_modules = session_state.get("_loaded_tab_modules")
        if not isinstance(loaded_tab_modules, set):
            loaded_tab_modules = set()
            session_state["_loaded_tab_modules"] = loaded_tab_modules

        for tab_container, module_path in zip(tabs, module_paths):
            if getattr(tab_container, "open", False):
                loaded_tab_modules.add(module_path)

        _render_tab_pages(
            tabs,
            module_paths,
            active_only=True,
            loaded_module_paths=loaded_tab_modules,
            import_module=import_module,
        )
    else:
        tabs = st.tabs(labels)
        _render_tab_pages(
            tabs,
            module_paths,
            active_only=False,
            import_module=import_module,
        )


def main() -> None:
    st.set_page_config(
        page_title="AI Face + Audio Recognition",
        page_icon="🎥",
        layout="wide",
    )
    st.set_option("client.showSidebarNavigation", False)

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
