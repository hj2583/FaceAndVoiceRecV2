from contextlib import nullcontext
from types import SimpleNamespace

import importlib
import sys


class _FakeNavigation:
    def __init__(self, events, pages):
        self._events = events
        self._pages = pages

    def run(self):
        self._events.append(("st.navigation.run", len(self._pages)))
        self._pages[0].render_fn()


class _FakePage:
    def __init__(self, render_fn, title, default):
        self.render_fn = render_fn
        self.title = title
        self.default = default


class _FakeStreamlit:
    def __init__(self, events):
        self._events = events

    def cache_resource(self, fn=None, **_kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    def set_page_config(self, **kwargs):
        self._events.append(("st.set_page_config", kwargs))

    def Page(self, render_fn, *, title, default=False):
        self._events.append(("st.Page", title, default))
        return _FakePage(render_fn, title=title, default=default)

    def navigation(self, pages, *, position="sidebar"):
        pages = list(pages)
        self._events.append(("st.navigation", len(pages), pages[0].title, position))
        return _FakeNavigation(self._events, pages)

    def title(self, value):
        self._events.append(("st.title", value))

    def caption(self, value):
        self._events.append(("st.caption", value))

    def tabs(self, labels):
        labels = list(labels)
        self._events.append(("st.tabs", labels))
        return [nullcontext() for _ in labels]


def test_importing_app_main_is_inert(monkeypatch):
    events = []
    fake_st = _FakeStreamlit(events)

    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.delitem(sys.modules, "app.main", raising=False)

    importlib.import_module("app.main")

    assert events == []


def test_initialize_application_database_caches_init_db(monkeypatch):
    monkeypatch.delitem(sys.modules, "app.main", raising=False)
    monkeypatch.delitem(sys.modules, "streamlit", raising=False)
    main = importlib.import_module("app.main")

    init_calls = []
    monkeypatch.setattr(main, "init_db", lambda: init_calls.append("init_db"))

    cache_clear = getattr(main.initialize_application_database, "clear", None)
    if cache_clear is not None:
        cache_clear()

    try:
        main.initialize_application_database()
        main.initialize_application_database()
    finally:
        cache_clear = getattr(main.initialize_application_database, "clear", None)
        if cache_clear is not None:
            cache_clear()

    assert init_calls == ["init_db"]


def test_main_renders_tabs_in_expected_order_with_startup_sequence(monkeypatch):
    main = importlib.import_module("app.main")

    events = []
    fake_st = _FakeStreamlit(events)

    monkeypatch.setattr(main, "st", fake_st)

    monkeypatch.setattr(main, "initialize_runtime_paths", lambda: events.append(("initialize_runtime_paths",)))
    monkeypatch.setattr(
        main,
        "initialize_application_database",
        lambda: events.append(("initialize_application_database",)),
    )
    page_modules = {
        "app.pages.realtime": SimpleNamespace(render=lambda: events.append(("render.realtime",))),
        "app.pages.video": SimpleNamespace(render=lambda: events.append(("render.video",))),
        "app.pages.face_database": SimpleNamespace(render=lambda: events.append(("render.face_database",))),
        "app.pages.audio_logs": SimpleNamespace(render=lambda: events.append(("render.audio_logs",))),
        "app.pages.attendance": SimpleNamespace(render=lambda: events.append(("render.attendance",))),
        "app.pages.transcripts": SimpleNamespace(render=lambda: events.append(("render.transcripts",))),
        "app.pages.voice_enrollment": SimpleNamespace(render=lambda: events.append(("render.voice_enrollment",))),
    }
    monkeypatch.setattr(main.importlib, "import_module", lambda module_path: page_modules[module_path])
    monkeypatch.setattr(
        main,
        "get_runtime_status",
        lambda: {"torch": "fake-torch", "face": "fake-face"},
    )

    main.main()

    assert events == [
        (
            "st.set_page_config",
            {
                "page_title": "AI Face + Audio Recognition",
                "page_icon": "🎥",
                "layout": "wide",
            },
        ),
        ("initialize_runtime_paths",),
        ("initialize_application_database",),
        ("st.Page", "AI Face + Active Speaker Recognition", True),
        ("st.navigation", 1, "AI Face + Active Speaker Recognition", "hidden"),
        ("st.navigation.run", 1),
        ("st.title", "🎥 AI Face + Active Speaker Recognition"),
        ("st.caption", "Audio models (PyTorch): fake-torch"),
        ("st.caption", "Face models (ONNX Runtime): fake-face"),
        (
            "st.tabs",
            [
                "📹 Realtime",
                "🎬 Video",
                "👤 Face Database",
                "🔊 Audio Logs",
                "✅ Attendance",
                "📝 Transcripts",
                "🎙️ Voice Enrollment",
            ],
        ),
        ("render.realtime",),
        ("render.video",),
        ("render.face_database",),
        ("render.audio_logs",),
        ("render.attendance",),
        ("render.transcripts",),
        ("render.voice_enrollment",),
    ]
