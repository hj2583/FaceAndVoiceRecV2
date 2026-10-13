import pandas as pd


def build_audio_log_frame(rows):
    frame = pd.DataFrame(rows)
    if "Track" in frame.columns:
        frame["Track"] = pd.array(frame["Track"], dtype="Int64")
    return frame