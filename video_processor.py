import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from audio_core import detect_speech_segments
from detection_core import detect_faces_tiled

from config import (
    ENABLE_TRANSCRIPTION,
    LIP_OPEN_THRESHOLD,
    OUTPUT_FPS_FALLBACK,

    RECOGNITION_INTERVAL,
    RECOGNITION_THRESHOLD,

    MIN_FACE_SIZE,
    MIN_RECOGNITION_FACE_SIZE,
    LANDMARK_MIN_FACE_SIZE,

    FACE_UPSCALE_THRESHOLD,
    FACE_UPSCALE_FACTOR,

    TRACKED_VIDEO_DIR,
    MAX_FACES,

    UNKNOWN_CONFIRMATIONS,
    UNKNOWN_MIN_QUALITY,
    UNKNOWN_FACES_DIR,
    NMS_IOU_THRESHOLD,
    TILE_GRID,
    TILE_OVERLAP,
    TILE_UPSCALE,
    TILED_DETECTION_INTERVAL,
)

from database import (
    log_audio,
    log_recognition,
)

from face_core import (
    FaceIndex,
    extract_embedding,
    register_unknown,
    face_quality,
)

from tracking import CentroidTracker


# ============================================================
# Recognition stability
# ============================================================

# If a previously recognized face temporarily fails recognition,
# keep the previous identity for a small number of attempts.
MAX_RECOGNITION_FAILURES = 3


# ============================================================
# FFmpeg
# ============================================================

def get_ffmpeg():
    from shutil import which

    path = which("ffmpeg")

    if not path:
        raise RuntimeError(
            "FFmpeg was not found in PATH. "
            "Install FFmpeg and add it to PATH."
        )

    return path


# ============================================================
# Extract audio from video
# ============================================================

def extract_audio_to_wav(video_path):
    ffmpeg = get_ffmpeg()

    temp = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )

    temp.close()

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        temp.name,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:

        logging.error(
            "FFmpeg audio extraction failed: %s",
            result.stderr[-2000:],
        )

        try:
            os.remove(temp.name)
        except OSError:
            pass

        return None

    return Path(temp.name)


# ============================================================
# Convert output video to browser-friendly H264
# ============================================================

def convert_h264(input_path, output_path):
    ffmpeg = get_ffmpeg()

    temp_path = str(output_path) + ".h264.mp4"

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        temp_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:

        logging.error(
            "FFmpeg H264 conversion failed: %s",
            result.stderr[-2000:],
        )

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return False

    os.replace(
        temp_path,
        output_path,
    )

    return True


# ============================================================
# MediaPipe Face Landmarker
# ============================================================

def create_face_landmarker():
    """
    Create MediaPipe FaceLandmarker.

    The face_landmarker.task model must exist in the same
    directory as this Python file.
    """

    model_path = (
        Path(__file__).resolve().parent
        / "face_landmarker.task"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe Face Landmarker model not found: "
            f"{model_path}"
        )

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(model_path)
    )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,

        running_mode=(
            mp.tasks.vision.RunningMode.VIDEO
        ),

        num_faces=MAX_FACES,

        min_face_detection_confidence=0.4,

        min_face_presence_confidence=0.4,

        min_tracking_confidence=0.5,

        output_face_blendshapes=False,

        output_facial_transformation_matrixes=False,
    )

    return (
        mp.tasks.vision.FaceLandmarker
        .create_from_options(options)
    )


# ============================================================
# Bounding box
# ============================================================

def bbox_from_landmarks(
    face_landmarks,
    frame_shape,
    padding=0.15,
):
    """
    Create a bounding box from MediaPipe face landmarks.
    """

    h, w = frame_shape[:2]

    xs = np.array(
        [lm.x for lm in face_landmarks]
    )

    ys = np.array(
        [lm.y for lm in face_landmarks]
    )

    x0 = int(xs.min() * w)
    x1 = int(xs.max() * w)

    y0 = int(ys.min() * h)
    y1 = int(ys.max() * h)

    px = int(
        (x1 - x0) * padding
    )

    py = int(
        (y1 - y0) * padding
    )

    return (
        max(0, x0 - px),
        max(0, y0 - py),
        min(w, x1 + px),
        min(h, y1 + py),
    )


# ============================================================
# Lip opening ratio
# ============================================================

def lip_open_ratio(face_landmarks):
    """
    Calculate mouth opening ratio.

    MediaPipe face landmark indexes:

        13  = upper inner lip
        14  = lower inner lip
        61  = left mouth corner
        291 = right mouth corner
    """

    p13 = face_landmarks[13]
    p14 = face_landmarks[14]

    p61 = face_landmarks[61]
    p291 = face_landmarks[291]

    vertical = abs(
        p14.y - p13.y
    )

    mouth_width = max(
        abs(p291.x - p61.x),
        1e-6,
    )

    return float(
        vertical / mouth_width
    )


# ============================================================
# Prepare recognition crop
# ============================================================

def prepare_recognition_crop(crop):
    """
    Prepare a face crop for ArcFace.

    Small faces are enlarged before embedding extraction.

    IMPORTANT:
    Upscaling does not create new facial information.
    It only helps the embedding model process the crop
    more consistently.
    """

    if crop is None or crop.size == 0:
        return None

    face_height, face_width = crop.shape[:2]

    # --------------------------------------------------------
    # Do not attempt recognition on extremely tiny faces.
    # --------------------------------------------------------

    if (
        face_width < MIN_RECOGNITION_FACE_SIZE
        or face_height < MIN_RECOGNITION_FACE_SIZE
    ):
        return None

    recognition_crop = crop

    # --------------------------------------------------------
    # Upscale smaller but usable faces.
    # --------------------------------------------------------

    if (
        face_width < FACE_UPSCALE_THRESHOLD
        or face_height < FACE_UPSCALE_THRESHOLD
    ):

        recognition_crop = cv2.resize(
            crop,
            None,
            fx=FACE_UPSCALE_FACTOR,
            fy=FACE_UPSCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC,
        )

    return recognition_crop


def _box_iou(box_a, box_b):
    """Return IoU for two ``(x0, y0, x1, y1)`` boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


# ============================================================
# Main video processing pipeline
# ============================================================

def process_video_pipeline(
    input_path,
    output_path,
    log_path,
    progress_callback=None,
):

    input_path = Path(input_path)
    output_path = Path(output_path)
    log_path = Path(log_path)

    # ========================================================
    # Open video
    # ========================================================

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {input_path}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = (
        cap.get(cv2.CAP_PROP_FPS)
        or OUTPUT_FPS_FALLBACK
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # ========================================================
    # Temporary video
    # ========================================================

    temp_output = output_path.with_suffix(
        ".raw.mp4"
    )

    writer = cv2.VideoWriter(
        str(temp_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create temporary video writer."
        )

    # ========================================================
    # Audio analysis
    # ========================================================

    wav_path = extract_audio_to_wav(
        input_path
    )

    speech_segments = []

    if wav_path:

        try:

            speech_segments = (
                detect_speech_segments(
                    wav_path
                )
            )

        except Exception:

            logging.exception(
                "Video audio VAD failed"
            )

        finally:

            try:
                wav_path.unlink()
            except OSError:
                pass

    # ========================================================
    # MediaPipe
    # ========================================================

    face_landmarker = (
        create_face_landmarker()
    )

    # ========================================================
    # Face recognition / tracking
    # ========================================================

    index = FaceIndex()

    tracker = CentroidTracker()

    frame_no = 0

    recognition_logs = []

    active_speech_logs = []
    last_tiled_detections = []

    # ========================================================
    # Processing
    # ========================================================

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_no += 1

            timestamp = (
                frame_no / fps
            )

            # =================================================
            # Tiled RetinaFace detection
            # =================================================

            if (
                not last_tiled_detections
                or frame_no % TILED_DETECTION_INTERVAL == 1
            ):
                last_tiled_detections = detect_faces_tiled(
                    frame,
                    tile_grid=TILE_GRID,
                    overlap_ratio=TILE_OVERLAP,
                    upscale_factor=TILE_UPSCALE,
                    nms_iou_threshold=NMS_IOU_THRESHOLD,
                )

            tiled_detections = last_tiled_detections

            # =================================================
            # BGR → RGB for landmark enrichment
            # =================================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            # Convert timestamp to milliseconds for MediaPipe
            timestamp_ms = int(timestamp * 1000)

            detections = []
            landmark_call_offset = 0
            for x, y, box_width, box_height, _confidence in tiled_detections:
                bbox = (
                    int(x),
                    int(y),
                    int(x + box_width),
                    int(y + box_height),
                )
                x0, y0, x1, y1 = bbox
                face_width = x1 - x0
                face_height = y1 - y0
                if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
                    continue

                landmarks = None
                lip_open = None
                if (
                    face_width >= LANDMARK_MIN_FACE_SIZE
                    and face_height >= LANDMARK_MIN_FACE_SIZE
                ):
                    crop_rgb = rgb[y0:y1, x0:x1]
                    if crop_rgb.size:
                        crop_mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=crop_rgb,
                        )
                        landmark_call_offset += 1
                        crop_result = face_landmarker.detect_for_video(
                            crop_mp_image,
                            timestamp_ms + landmark_call_offset,
                        )
                        if crop_result.face_landmarks:
                            landmarks = crop_result.face_landmarks[0]
                            lip_open = lip_open_ratio(landmarks)

                detections.append(
                    {
                        "bbox": bbox,
                        "landmarks": landmarks,
                        "lip_open": lip_open,
                    }
                )

            # =================================================
            # Tracking
            # =================================================

            assignments = tracker.update(
                detections,
                frame_no,
            )

            # =================================================
            # Audio speech state
            # =================================================

            audio_is_speech = any(
                start <= timestamp <= end
                for start, end in speech_segments
            )

            speaking_candidates = []

            # =================================================
            # Process tracked faces
            # =================================================

            for detection, track in assignments:

                # ------------------------------------------------
                # Initialize recognition state
                # ------------------------------------------------

                if not hasattr(
                    track,
                    "recognition_failures",
                ):
                    track.recognition_failures = 0

                if not hasattr(
                    track,
                    "last_good_embedding",
                ):
                    track.last_good_embedding = None

                if not hasattr(
                    track,
                    "last_good_similarity",
                ):
                    track.last_good_similarity = 0.0

                if not hasattr(
                    track,
                    "unknown_confirmations",
                ):
                    track.unknown_confirmations = 0

                if not hasattr(
                    track,
                    "registered_unknown",
                ):
                    track.registered_unknown = False

                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                x0, y0, x1, y1 = (
                    detection["bbox"]
                )

                face_width = x1 - x0
                face_height = y1 - y0

                crop = frame[
                    y0:y1,
                    x0:x1,
                ]

                if crop.size == 0:
                    continue

                # =================================================
                # Face recognition
                # =================================================

                can_recognize = (
                    face_width
                    >= MIN_RECOGNITION_FACE_SIZE
                    and
                    face_height
                    >= MIN_RECOGNITION_FACE_SIZE
                )

                should_recognize = (
                    track.embedding is None
                    or
                    frame_no
                    % RECOGNITION_INTERVAL
                    == 0
                )

                if (
                    can_recognize
                    and should_recognize
                ):

                    recognition_crop = (
                        prepare_recognition_crop(
                            crop
                        )
                    )

                    if recognition_crop is not None:

                        embedding = (
                            extract_embedding(
                                recognition_crop
                            )
                        )

                        if embedding is not None:

                            match = (
                                index.search(
                                    embedding
                                )
                            )

                            similarity = (
                                match.get(
                                    "similarity",
                                    0.0,
                                )
                                if match
                                else 0.0
                            )

                            # =================================================
                            # Valid recognition
                            # =================================================

                            if (
                                match
                                and
                                similarity
                                >= RECOGNITION_THRESHOLD
                            ):

                                from database import (
                                    get_conn,
                                )

                                with get_conn() as conn:

                                    row = conn.execute(
                                        """
                                        SELECT
                                            p.person_id,
                                            p.name
                                        FROM face_embeddings fe
                                        JOIN persons p
                                            ON p.person_id =
                                               fe.person_id
                                        WHERE fe.embedding_id = ?
                                        """,
                                        (
                                            match[
                                                "embedding_id"
                                            ],
                                        ),
                                    ).fetchone()

                                if row:

                                    track.person_id = (
                                        int(row[0])
                                    )

                                    track.person_name = (
                                        row[1]
                                    )

                                    track.confidence = (
                                        float(
                                            similarity
                                        )
                                    )

                                    track.embedding = (
                                        embedding
                                    )

                                    track.last_good_embedding = (
                                        embedding
                                    )

                                    track.last_good_similarity = (
                                        float(
                                            similarity
                                        )
                                    )

                                    track.recognition_failures = 0

                                    track.unknown_confirmations = 0

                            # =================================================
                            # Recognition failed
                            # =================================================

                            else:

                                track.recognition_failures += 1

                                # ------------------------------------------------
                                # Already recognized:
                                # temporarily keep identity.
                                # ------------------------------------------------

                                if (
                                    track.person_id
                                    is not None
                                ):

                                    if (
                                        track.recognition_failures
                                        <= MAX_RECOGNITION_FAILURES
                                    ):

                                        if (
                                            track.last_good_embedding
                                            is not None
                                        ):

                                            track.embedding = (
                                                track.last_good_embedding
                                            )

                                    logging.debug(
                                        "Temporary recognition "
                                        "failure: track=%s "
                                        "person=%s "
                                        "failure=%s/%s",
                                        track.track_id,
                                        track.person_name,
                                        track.recognition_failures,
                                        MAX_RECOGNITION_FAILURES,
                                    )

                                # ------------------------------------------------
                                # Never recognized:
                                # remain Unknown.
                                # ------------------------------------------------

                                else:

                                    track.person_name = (
                                        "Unknown"
                                    )

                                    track.confidence = 0.0

                                    track.embedding = (
                                        embedding
                                    )

                # =================================================
                # Unknown face registration
                # =================================================

                if (
                    track.person_id is None
                    and
                    track.person_name == "Unknown"
                    and
                    track.embedding is not None
                ):

                    track.unknown_confirmations += 1

                    if (
                        track.unknown_confirmations
                        >= UNKNOWN_CONFIRMATIONS
                        and
                        not track.registered_unknown
                    ):

                        quality = face_quality(
                            crop
                        )

                        if (
                            quality
                            >= UNKNOWN_MIN_QUALITY
                        ):

                            UNKNOWN_FACES_DIR.mkdir(
                                parents=True,
                                exist_ok=True,
                            )

                            unknown_label = (
                                f"track_{track.track_id}"
                            )

                            image_path = (
                                UNKNOWN_FACES_DIR
                                / f"{unknown_label}.jpg"
                            )

                            embedding_path = (
                                UNKNOWN_FACES_DIR
                                / f"{unknown_label}.npy"
                            )

                            success = cv2.imwrite(
                                str(image_path),
                                crop,
                            )

                            if success:

                                np.save(
                                    embedding_path,
                                    track.embedding,
                                )

                                try:

                                    register_unknown(
                                        label=unknown_label,
                                        embedding=track.embedding,
                                        image_path=image_path,
                                    )

                                    track.registered_unknown = (
                                        True
                                    )

                                except Exception:

                                    logging.exception(
                                        "Failed to register "
                                        "unknown face"
                                    )

                            else:

                                logging.warning(
                                    "Could not save unknown "
                                    "face image: %s",
                                    image_path,
                                )

                # =================================================
                # Recognition log
                # =================================================

                if (
                    track.person_id
                    is not None
                ):

                    recognition_logs.append(
                        {
                            "timestamp_sec": round(
                                timestamp,
                                2,
                            ),
                            "person_name": (
                                track.person_name
                            ),
                            "confidence": round(
                                track.confidence,
                                4,
                            ),
                        }
                    )

                # =================================================
                # Lip movement
                # =================================================

                track.lip_open = (
                    detection["lip_open"]
                )

                # =================================================
                # Active speaker candidates
                # =================================================

                if (
                    audio_is_speech
                    and
                    track.lip_open is not None
                    and
                    track.lip_open
                    >= LIP_OPEN_THRESHOLD
                ):

                    speaking_candidates.append(
                        track
                    )

                # =================================================
                # Draw bounding box
                # =================================================

                unknown = (
                    track.person_name
                    == "Unknown"
                )

                color = (
                    (0, 0, 255)
                    if unknown
                    else (0, 255, 0)
                )

                label = (
                    track.person_name
                    or "Recognition Pending"
                )

                # ------------------------------------------------
                # Indicate when face is too small for recognition.
                # ------------------------------------------------

                if not can_recognize:

                    if (
                        track.person_id
                        is None
                    ):

                        label = (
                            "Recognition Pending"
                        )

                if (
                    track
                    in speaking_candidates
                ):

                    label += " [Speaking]"

                cv2.rectangle(
                    frame,
                    (x0, y0),
                    (x1, y1),
                    color,
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x0,
                        max(
                            20,
                            y0 - 10,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # =====================================================
            # Select ONE active speaker
            # =====================================================

            if (
                audio_is_speech
                and
                speaking_candidates
            ):

                speaker = max(
                    speaking_candidates,
                    key=lambda t:
                        t.lip_open,
                )

                speaker.speech_frames += 1
                speaker.silent_frames = 0

                if (
                    speaker.speech_frames
                    >= 2
                ):

                    start = max(
                        0.0,
                        timestamp - 0.1,
                    )

                    end = timestamp

                    if (
                        speaker.person_id
                        is not None
                    ):

                        log_audio(
                            start,
                            end,
                            speaker.person_id,
                            speaker.person_name,
                            speaker.confidence,
                            "video",
                        )

                        active_speech_logs.append(
                            {
                                "start_time": round(
                                    start,
                                    2,
                                ),
                                "end_time": round(
                                    end,
                                    2,
                                ),
                                "person_name": (
                                    speaker.person_name
                                ),
                                "confidence": round(
                                    speaker.confidence,
                                    4,
                                ),
                                "source": "video",
                            }
                        )

            else:

                for track in (
                    tracker.tracks.values()
                ):

                    track.silent_frames += 1

                    if (
                        track.silent_frames
                        >= 4
                    ):

                        track.speech_frames = 0

            # =================================================
            # Write frame
            # =================================================

            writer.write(frame)

            # =================================================
            # Progress
            # =================================================

            if (
                progress_callback
                and total_frames
            ):

                progress_callback(
                    frame_no
                    / total_frames
                )

    finally:

        cap.release()

        writer.release()

        face_landmarker.close()

    # ==========================================================
    # Convert to H264
    # ==========================================================

    if not convert_h264(
        temp_output,
        output_path,
    ):

        os.replace(
            temp_output,
            output_path,
        )

    # ==========================================================
    # Save logs
    # ==========================================================

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "recognition": (
                    recognition_logs
                ),
                "speech": (
                    active_speech_logs
                ),
            },
            f,
            indent=2,
        )

    return True