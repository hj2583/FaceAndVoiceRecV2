from types import SimpleNamespace

import ffmpeg_utils


def test_find_ffmpeg_prefers_system_path(monkeypatch):
    monkeypatch.setattr(ffmpeg_utils.shutil, "which", lambda _name: "C:/tools/ffmpeg.exe")

    assert ffmpeg_utils.find_ffmpeg() == "C:/tools/ffmpeg.exe"


def test_find_ffmpeg_uses_imageio_binary_when_system_path_is_missing(monkeypatch):
    bundled_executable = "C:/venv/site-packages/imageio_ffmpeg/binaries/ffmpeg.exe"
    monkeypatch.setattr(ffmpeg_utils.shutil, "which", lambda _name: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: bundled_executable),
    )

    assert ffmpeg_utils.find_ffmpeg() == bundled_executable