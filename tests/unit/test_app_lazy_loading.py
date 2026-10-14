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
    rendered = []

    def import_module(module_path):
        return SimpleNamespace(render=lambda: rendered.append(module_path))

    main._render_tab_pages(tabs, module_paths, active_only=True, import_module=import_module)

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
        def title(self, value):
            events.append(("title", value))

        def caption(self, value):
            events.append(("caption", value))

        def tabs(self, labels, *, on_change=None, key=None):
            events.append(("tabs", list(labels), on_change, key))
            return [FakeTab(index == 1) for index, _ in enumerate(labels)]

    rendered = []

    def import_module(module_path):
        return SimpleNamespace(render=lambda: rendered.append(module_path))

    monkeypatch.setattr(main, "st", FakeStatefulStreamlit())
    monkeypatch.setattr(main, "get_runtime_status", lambda: {"torch": "ok", "face": "ok"})
    monkeypatch.setattr(main.importlib, "import_module", import_module)

    main.render_application()

    assert events[0] == ("title", "🎥 AI Face + Active Speaker Recognition")
    assert events[3][0] == "tabs"
    assert events[3][2] == "rerun"
    assert events[3][3] == "main-tabs"
    assert rendered == ["app.pages.video"]


def test_render_application_fallback_renders_all_pages(monkeypatch):
    main = _load_main_with_fake_streamlit(monkeypatch)

    events = []

    class FakeLegacyStreamlit:
        def title(self, value):
            events.append(("title", value))

        def caption(self, value):
            events.append(("caption", value))

        def tabs(self, labels):
            events.append(("tabs", list(labels)))
            return [FakeTab(False) for _ in labels]

    rendered = []

    def import_module(module_path):
        return SimpleNamespace(render=lambda: rendered.append(module_path))

    monkeypatch.setattr(main, "st", FakeLegacyStreamlit())
    monkeypatch.setattr(main, "get_runtime_status", lambda: {"torch": "ok", "face": "ok"})
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
