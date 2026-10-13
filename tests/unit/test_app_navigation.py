from contextlib import nullcontext

import importlib


class _FakeStreamlit:
    def __init__(self):
        self.calls = []

    def set_page_config(self, **kwargs):
        self.calls.append(("set_page_config", kwargs))

    def title(self, value):
        self.calls.append(("title", value))

    def caption(self, value):
        self.calls.append(("caption", value))

    def tabs(self, labels):
        labels = list(labels)
        self.calls.append(("tabs", labels))
        return [nullcontext() for _ in labels]


def test_main_renders_tabs_in_expected_order_with_startup_sequence(monkeypatch):
    main = importlib.import_module("app.main")

    fake_st = _FakeStreamlit()
    calls = []

    monkeypatch.setattr(main, "st", fake_st)

    monkeypatch.setattr(main, "initialize_runtime_paths", lambda: calls.append("init_paths"))
    monkeypatch.setattr(main, "init_db", lambda: calls.append("init_db"))
    monkeypatch.setattr(main.realtime, "render", lambda: calls.append("realtime"))
    monkeypatch.setattr(main.video, "render", lambda: calls.append("video"))
    monkeypatch.setattr(main.face_database, "render", lambda: calls.append("face_database"))
    monkeypatch.setattr(main.audio_logs, "render", lambda: calls.append("audio_logs"))
    monkeypatch.setattr(main.attendance, "render", lambda: calls.append("attendance"))
    monkeypatch.setattr(main.transcripts, "render", lambda: calls.append("transcripts"))
    monkeypatch.setattr(main.voice_enrollment, "render", lambda: calls.append("voice_enrollment"))
    monkeypatch.setattr(
        main,
        "get_runtime_status",
        lambda: {"torch": "fake-torch", "face": "fake-face"},
    )

    main.main()

    assert calls == [
        "init_paths",
        "init_db",
        "realtime",
        "video",
        "face_database",
        "audio_logs",
        "attendance",
        "transcripts",
        "voice_enrollment",
    ]

    assert fake_st.calls[0][0] == "set_page_config"
    assert fake_st.calls[1] == ("title", "🎥 AI Face + Active Speaker Recognition")
    assert fake_st.calls[2] == ("caption", "Audio models (PyTorch): fake-torch")
    assert fake_st.calls[3] == ("caption", "Face models (ONNX Runtime): fake-face")
    assert fake_st.calls[4] == (
        "tabs",
        [
            "📹 Realtime",
            "🎬 Video",
            "👤 Face Database",
            "🔊 Audio Logs",
            "✅ Attendance",
            "📝 Transcripts",
            "🎙️ Voice Enrollment",
        ],
    )
