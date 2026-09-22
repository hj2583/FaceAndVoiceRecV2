import logging
import threading
import wave

import numpy as np
import torch

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from silero_vad import load_silero_vad
except Exception:
    load_silero_vad = None


# ============================================================
# Audio configuration
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

# Silero VAD requires exactly 512 samples per call at 16kHz (32 ms).
FRAME_MS = 32

FRAME_SAMPLES = int(
    SAMPLE_RATE * FRAME_MS / 1000
)

FRAME_BYTES = (
    FRAME_SAMPLES * SAMPLE_WIDTH
)


# ============================================================
# VAD configuration
# ============================================================

# Silero VAD outputs a speech probability in [0, 1] per frame.
# 0.5 is the model's recommended default decision threshold.
VAD_SPEECH_THRESHOLD = 0.5


# ------------------------------------------------------------
# Realtime smoothing
# ------------------------------------------------------------

# Number of consecutive speech frames required before
# realtime mode changes from silent -> speaking.
#
# 3 frames × 32 ms = 96 ms
REALTIME_SPEECH_CONFIRM_FRAMES = 3


# Number of consecutive silent frames required before
# realtime mode changes from speaking -> silent.
#
# 8 frames × 32 ms = 256 ms
#
# This prevents tiny pauses between words from causing
# speech state to flicker.
REALTIME_SILENCE_CONFIRM_FRAMES = 8


# ------------------------------------------------------------
# Offline speech segmentation
# ------------------------------------------------------------

# Minimum duration for a valid speech segment.
MIN_SPEECH_SEGMENT_MS = 300


# Extra audio before speech starts.
SPEECH_START_PADDING_MS = 200


# Extra audio after speech ends.
SPEECH_END_PADDING_MS = 200


# Maximum gap allowed between two speech segments before
# they are merged.
SPEECH_MERGE_GAP_MS = 250


def check_audio_dependencies():
    if sd is None:
        raise RuntimeError("sounddevice is not installed.")
    if load_silero_vad is None:
        raise RuntimeError("silero-vad is not installed.")


def _speech_probability(model, frame_bytes, sample_rate=SAMPLE_RATE):
    """
    Run Silero VAD on one raw PCM16 frame and return a speech probability.

    `model` is expected to be callable as `model(tensor, sample_rate)`,
    matching Silero VAD's `load_silero_vad()` return value.
    """

    audio = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if audio.size < FRAME_SAMPLES:
        audio = np.pad(audio, (0, FRAME_SAMPLES - audio.size))

    with torch.no_grad():
        result = model(torch.from_numpy(audio), sample_rate)

    return float(result.item()) if hasattr(result, "item") else float(result)


class RealtimeVAD:
    """
    Local microphone VAD with speech-state smoothing.

    The callback receives a boolean:
        True  -> currently speaking
        False -> currently silent

    Short speech/noise fluctuations are filtered so the
    callback does not rapidly switch between True and False.
    """

    def __init__(
        self,
        callback=None,
        threshold=VAD_SPEECH_THRESHOLD,
    ):
        self.callback = callback
        self.threshold = threshold

        self.stop_event = threading.Event()
        self.thread = None

        # Current smoothed speech state.
        self.speaking = False

        # Consecutive frame counters.
        self.speech_frames = 0
        self.silence_frames = 0

        # Startup handshake for async worker.
        self.startup_event = threading.Event()
        self.startup_succeeded = False
        self.available = False
        self.startup_error = None

    def start(self, startup_timeout=2.0):
        check_audio_dependencies()

        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError(
                "Realtime VAD worker is already running"
            )

        # Reset state in case the same object is started again.
        self.stop_event.clear()
        self.startup_event.clear()
        self.startup_succeeded = False
        self.available = False
        self.startup_error = None
        self.speaking = False
        self.speech_frames = 0
        self.silence_frames = 0

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self.thread.start()

        if not self.startup_event.wait(timeout=startup_timeout):
            self.stop()
            raise RuntimeError("Microphone startup timed out")

        if not self.startup_succeeded:
            message = (
                str(self.startup_error)
                if self.startup_error
                else "Microphone startup failed"
            )
            self.stop()
            raise RuntimeError(message) from self.startup_error

    def stop(self):
        self.stop_event.set()

        if self.thread:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                logging.warning(
                    "Realtime VAD worker did not stop within timeout"
                )
            else:
                self.thread = None

    def _emit(self, state):
        """
        Send the current smoothed speech state to the callback.
        """

        if self.callback:
            try:
                self.callback(bool(state))
            except Exception:
                logging.exception(
                    "Realtime VAD callback failed"
                )

    def _update_speech_state(self, raw_speech):
        """
        Convert raw Silero VAD output into a stable speech state.

        This prevents:
            True -> False -> True -> False

        from happening because of very short pauses or noise.
        """

        if raw_speech:

            self.speech_frames += 1
            self.silence_frames = 0

            # Only switch to speaking after enough
            # consecutive speech frames.
            if (
                not self.speaking
                and self.speech_frames
                >= REALTIME_SPEECH_CONFIRM_FRAMES
            ):
                self.speaking = True

        else:

            self.silence_frames += 1
            self.speech_frames = 0

            # Only switch to silence after enough
            # consecutive silent frames.
            if (
                self.speaking
                and self.silence_frames
                >= REALTIME_SILENCE_CONFIRM_FRAMES
            ):
                self.speaking = False

        return self.speaking

    def _run(self):
        stream = None

        try:

            model = load_silero_vad()

            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
            )

            stream.start()

            self.available = True
            self.startup_succeeded = True
            self.startup_event.set()

            while not self.stop_event.is_set():

                data, _overflowed = stream.read(
                    FRAME_SAMPLES,
                )

                raw_speech = _speech_probability(
                    model,
                    bytes(data),
                ) >= self.threshold

                speech = self._update_speech_state(
                    bool(raw_speech)
                )

                self._emit(speech)

        except Exception as error:
            if not self.startup_succeeded:
                self.startup_error = error
            logging.exception(
                "Realtime microphone VAD failed"
            )

        finally:

            self.available = False
            if self.speaking:
                self.speaking = False
                self._emit(False)
            if not self.startup_event.is_set():
                self.startup_event.set()

            if stream:

                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    logging.exception(
                        "Failed to close microphone stream"
                    )

def read_wav_pcm(wav_path):
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if rate != SAMPLE_RATE or channels != 1 or width != SAMPLE_WIDTH:
        raise ValueError(
            f"WAV must be 16kHz mono 16-bit, got "
            f"{rate}Hz/{channels}ch/{width * 8}bit"
        )

    return frames


def detect_speech_segments(
    wav_path,
    min_segment_ms=MIN_SPEECH_SEGMENT_MS,
    padding_ms=SPEECH_START_PADDING_MS,
    threshold=VAD_SPEECH_THRESHOLD,
):
    """
    Detect clean speech segments from a WAV file.

    Returns:
        [
            (start_seconds, end_seconds),
            ...
        ]

    Processing stages:

        WAV
          ↓
        Silero VAD
          ↓
        speech/silence state smoothing
          ↓
        minimum segment filtering
          ↓
        padding
          ↓
        nearby segment merging
    """

    check_audio_dependencies()

    pcm = read_wav_pcm(
        wav_path
    )

    model = load_silero_vad()

    bytes_per_frame = FRAME_BYTES

    total_frames = (
        len(pcm) // bytes_per_frame
    )

    if total_frames == 0:
        return []

    # ========================================================
    # Step 1: Raw VAD
    # ========================================================

    states = []

    for i in range(total_frames):

        frame = pcm[
            i * bytes_per_frame:
            (i + 1) * bytes_per_frame
        ]

        speech = _speech_probability(
            model,
            frame,
        ) >= threshold

        states.append(
            bool(speech)
        )

    # ========================================================
    # Step 2: Convert raw VAD states into segments
    # ========================================================

    segments = []

    start = None

    # Number of consecutive silent frames allowed inside
    # the same speech segment.
    silence_tolerance_frames = max(
        1,
        round(
            SPEECH_END_PADDING_MS
            / FRAME_MS
        ),
    )

    silence_frames = 0

    for i, speech in enumerate(states):

        timestamp = (
            i * FRAME_MS / 1000.0
        )

        if speech:

            # ------------------------------------------------
            # Speech detected
            # ------------------------------------------------

            if start is None:

                start = max(
                    0.0,
                    timestamp
                    - padding_ms / 1000.0,
                )

            silence_frames = 0

        else:

            # ------------------------------------------------
            # Silence detected
            # ------------------------------------------------

            if start is not None:

                silence_frames += 1

                # Don't immediately end the segment.
                #
                # This allows short pauses between words
                # to remain part of the same speech segment.
                if (
                    silence_frames
                    < silence_tolerance_frames
                ):
                    continue

                end = (
                    timestamp
                    + padding_ms / 1000.0
                )

                duration_ms = (
                    end - start
                ) * 1000.0

                if (
                    duration_ms
                    >= min_segment_ms
                ):

                    segments.append(
                        (
                            start,
                            end,
                        )
                    )

                start = None
                silence_frames = 0

    # ========================================================
    # Step 3: Handle speech continuing until EOF
    # ========================================================

    if start is not None:

        end = (
            total_frames
            * FRAME_MS
            / 1000.0
        )

        end += (
            padding_ms / 1000.0
        )

        duration_ms = (
            end - start
        ) * 1000.0

        if (
            duration_ms
            >= min_segment_ms
        ):

            segments.append(
                (
                    start,
                    end,
                )
            )

    # ========================================================
    # Step 4: Merge nearby speech segments
    # ========================================================

    return _merge_segments(
        segments,
        gap=(
            SPEECH_MERGE_GAP_MS
            / 1000.0
        ),
    )

def _merge_segments(
    segments,
    gap=SPEECH_MERGE_GAP_MS / 1000.0,
):
    """
    Merge speech segments separated by a short gap.

    Example:

        (1.0, 2.0)
        (2.15, 3.0)

    becomes:

        (1.0, 3.0)

    when gap >= 0.15 seconds.
    """

    if not segments:
        return []

    # Sort by start time first.
    segments = sorted(
        segments,
        key=lambda x: x[0],
    )

    merged = [
        [
            float(segments[0][0]),
            float(segments[0][1]),
        ]
    ]

    for start, end in segments[1:]:

        start = float(start)
        end = float(end)

        previous_end = merged[-1][1]

        if (
            start - previous_end
            <= gap
        ):

            merged[-1][1] = max(
                previous_end,
                end,
            )

        else:

            merged.append(
                [
                    start,
                    end,
                ]
            )

    return [
        (
            round(start, 3),
            round(end, 3),
        )
        for start, end in merged
    ]

def audio_rms_from_pcm(pcm_bytes):
    if not pcm_bytes:
        return 0.0
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio * audio)))
