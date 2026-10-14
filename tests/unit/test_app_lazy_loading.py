import importlib
import sys
from types import SimpleNamespace


class FakeTab:
    def __init__(self, opened: bool) -> None:
        self.open = opened

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _ImportFakeStreamlit:
    def cache_resource(self, fn=None, **_kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn


def _load_main_with_fake_streamlit(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", _ImportFakeStreamlit())
    monkeypatch.delitem(sys.modules, "app.main", raising=False)
    return importlib.import_module("app.main")


def test_main_import_does_not_bind_runtime_status_symbol(monkeypatch):
    monkeypatch.delitem(sys.modules, "app.ui.status", raising=False)
    main = _load_main_with_fake_streamlit(monkeypatch)

    assert callable(main.get_runtime_status)
    assert "app.ui.status" not in sys.modules


def test_stateful_tabs_import_only_the_open_page(monkeypatch):
    main = _load_main_with_fake_streamlit(monkeypatch)

    selected = "app.pages.video"
    module_paths = (
        "app.pages.realtime",
        "app.pages.video",
        "app.pages.face_database",
        "app.pages.audio_logs",
        "app.pages.attendance",
        "app.pages.transcripts",
        "app.pages.voice_enrollment",
    )
    tabs = [FakeTab(module_path == selected) for module_path in module_paths]
    imported = []
    rendered = []

    def import_module(module_path):
        imported.append(module_path)
        return SimpleNamespace(render=lambda: rendered.append(module_path))

    main._render_tab_pages(tabs, module_paths, active_only=True, import_module=import_module)

    assert imported == [selected]
    assert rendered == [selected]


def test_fallback_tabs_render_all_pages(monkeypatch):
    main = _load_main_with_fake_streamlit(monkeypatch)

    module_paths = (
        "app.pages.realtime",
        "app.pages.video",
        "app.pages.face_database",
        "app.pages.audio_logs",
        "app.pages.attendance",
        "app.pages.transcripts",
        "app.pages.voice_enrollment",
    )
    tabs = [FakeTab(False) for _ in module_paths]
    rendered = []

    def import_module(module_path):
        return SimpleNamespace(render=lambda: rendered.append(module_path))

    main._render_tab_pages(tabs, module_paths, active_only=False, import_module=import_module)

    assert rendered == list(module_paths)


def test_supports_stateful_tabs_detects_new_signature(monkeypatch):
    main = _load_main_with_fake_streamlit(monkeypatch)

    def stateful_tabs(labels, *, on_change=None, key=None):
        return labels, on_change, key

    assert main.supports_stateful_tabs(stateful_tabs) is True


def test_supports_stateful_tabs_detects_legacy_signature(monkeypatch):
    main = _load_main_with_fake_streamlit(monkeypatch)

    def legacy_tabs(labels):
        return labels

    assert main.supports_stateful_tabs(legacy_tabs) is False


def test_render_application_stateful_tabs_import_only_open(monkeypatch):
    main = _load_main_with_fake_streamlit(monkeypatch)

    events = []

    class FakeStatefulStreamlit:
        def __init__(self):
            self.session_state = {}

        def title(self, value):
            events.append(("title", value))

        def caption(self, value):
            events.append(("caption", value))

        def button(self, _label):
            return False

        def tabs(self, labels, *, on_change=None, key=None):
            events.append(("tabs", list(labels), on_change, key))
            return [FakeTab(index == 1) for index, _ in enumerate(labels)]

    imported = []
    rendered = []

    def import_module(module_path):
        if module_path == "app.ui.status":
            raise AssertionError("runtime status should not load before click")
        imported.append(module_path)
        return SimpleNamespace(render=lambda: rendered.append(module_path))

    monkeypatch.setattr(main, "st", FakeStatefulStreamlit())
    monkeypatch.setattr(main.importlib, "import_module", import_module)

    main.render_application()

    assert events[0] == ("title", "🎥 AI Face + Active Speaker Recognition")
    assert events[1][0] == "tabs"
    assert events[1][2] == "rerun"
    assert events[1][3] == "main-tabs"
    assert all(event[0] != "caption" for event in events)
    assert imported == ["app.pages.video"]
    assert rendered == ["app.pages.video"]


def test_render_application_without_button_api_does_not_probe_runtime(monkeypatch):
    main = _load_main_with_fake_streamlit(monkeypatch)

    class FakeStatefulStreamlit:
        def __init__(self):
            self.session_state = {}
            self.captions = []

        def title(self, _value):
            pass

        def caption(self, value):
            self.captions.append(value)

        def tabs(self, labels, *, on_change=None, key=None):
            return [FakeTab(index == 1) for index, _ in enumerate(labels)]

    fake_st = FakeStatefulStreamlit()

    def import_module(module_path):
        if module_path == "app.ui.status":
            raise AssertionError("runtime status should not load without a button click")
        return SimpleNamespace(render=lambda: None)

    monkeypatch.setattr(main, "st", fake_st)
    monkeypatch.setattr(main.importlib, "import_module", import_module)

    main.render_application()

    assert fake_st.session_state == {}
    assert fake_st.captions == []


def test_render_application_fallback_renders_all_pages(monkeypatch):
    main = _load_main_with_fake_streamlit(monkeypatch)

    events = []

    class FakeLegacyStreamlit:
        def __init__(self):
            self.session_state = {}

        def title(self, value):
            events.append(("title", value))

        def caption(self, value):
            events.append(("caption", value))

        def button(self, _label):
            return False

        def tabs(self, labels):
            events.append(("tabs", list(labels)))
            return [FakeTab(False) for _ in labels]

    rendered = []

    def import_module(module_path):
        if module_path == "app.ui.status":
            raise AssertionError("runtime status should not load before click")
        return SimpleNamespace(render=lambda: rendered.append(module_path))

    monkeypatch.setattr(main, "st", FakeLegacyStreamlit())
    monkeypatch.setattr(main.importlib, "import_module", import_module)

    main.render_application()

    assert rendered == [
        "app.pages.realtime",
        "app.pages.video",
        "app.pages.face_database",
        "app.pages.audio_logs",
        "app.pages.attendance",
        "app.pages.transcripts",
        "app.pages.voice_enrollment",
    ]
    assert all(event[0] != "caption" for event in events)


def test_render_application_runtime_check_clicks_once_and_caches(monkeypatch):
    main = _load_main_with_fake_streamlit(monkeypatch)

    imported = []
    rendered = []
    probe_calls = []

    class FakeRuntimeStatusModule:
        @staticmethod
        def get_runtime_status():
            probe_calls.append("called")
            return {"torch": "ok", "face": "ok"}

    class FakeStatefulStreamlit:
        def __init__(self):
            self.session_state = {}
            self._buttons = [True, False]
            self.captions = []
            self.button_labels = []

        def title(self, _value):
            pass

        def caption(self, value):
            self.captions.append(value)

        def button(self, label):
            self.button_labels.append(label)
            return self._buttons.pop(0)

        def tabs(self, labels, *, on_change=None, key=None):
            return [FakeTab(index == 1) for index, _ in enumerate(labels)]

    fake_st = FakeStatefulStreamlit()

    def import_module(module_path):
        imported.append(module_path)
        if module_path == "app.ui.status":
            return FakeRuntimeStatusModule()
        return SimpleNamespace(render=lambda: rendered.append(module_path))

    monkeypatch.setattr(main, "st", fake_st)
    monkeypatch.setattr(main.importlib, "import_module", import_module)

    main.render_application()
    main.render_application()

    assert probe_calls == ["called"]
    assert fake_st.session_state["runtime_status"] == {"torch": "ok", "face": "ok"}
    assert imported.count("app.ui.status") == 1
    assert fake_st.button_labels == ["Check runtime"]
    assert fake_st.captions.count("Audio models (PyTorch): ok") == 2
    assert fake_st.captions.count("Face models (ONNX Runtime): ok") == 2
    assert rendered == ["app.pages.video", "app.pages.video"]


def test_render_application_runtime_failure_then_retry_success(monkeypatch):
    main = _load_main_with_fake_streamlit(monkeypatch)

    imported = []
    rendered = []
    probe_calls = []
    status_results = [RuntimeError("probe failed"), {"torch": "ok", "face": "ok"}]

    class FakeRuntimeStatusModule:
        @staticmethod
        def get_runtime_status():
            probe_calls.append("called")
            result = status_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

    class FakeStatefulStreamlit:
        def __init__(self):
            self.session_state = {}
            self._buttons = [True, True]
            self.captions = []
            self.button_labels = []

        def title(self, _value):
            pass

        def caption(self, value):
            self.captions.append(value)

        def button(self, label):
            self.button_labels.append(label)
            return self._buttons.pop(0)

        def tabs(self, labels, *, on_change=None, key=None):
            return [FakeTab(index == 1) for index, _ in enumerate(labels)]

    fake_st = FakeStatefulStreamlit()

    def import_module(module_path):
        imported.append(module_path)
        if module_path == "app.ui.status":
            return FakeRuntimeStatusModule()
        return SimpleNamespace(render=lambda: rendered.append(module_path))

    monkeypatch.setattr(main, "st", fake_st)
    monkeypatch.setattr(main.importlib, "import_module", import_module)

    main.render_application()

    assert fake_st.session_state["runtime_status_error"] == "probe failed"
    assert "runtime_status" not in fake_st.session_state
    assert "Audio models (PyTorch): unavailable" in fake_st.captions
    assert "Face models (ONNX Runtime): unavailable" in fake_st.captions
    assert "Runtime status error: probe failed" in fake_st.captions
    assert fake_st.button_labels == ["Check runtime"]

    main.render_application()

    assert probe_calls == ["called", "called"]
    assert fake_st.session_state["runtime_status"] == {"torch": "ok", "face": "ok"}
    assert "runtime_status_error" not in fake_st.session_state
    assert imported.count("app.ui.status") == 2
    assert fake_st.button_labels == ["Check runtime", "Check runtime"]
    assert fake_st.captions.count("Audio models (PyTorch): ok") == 1
    assert fake_st.captions.count("Face models (ONNX Runtime): ok") == 1
    assert rendered == ["app.pages.video", "app.pages.video"]
