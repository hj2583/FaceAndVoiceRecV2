import argparse
import logging
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from audio_core import RealtimeVAD

from config import (
    LIP_OPEN_THRESHOLD,

    RECOGNITION_INTERVAL,
    RECOGNITION_THRESHOLD,

    MIN_FACE_SIZE,
    MIN_RECOGNITION_FACE_SIZE,

    FACE_UPSCALE_THRESHOLD,
    FACE_UPSCALE_FACTOR,

    MAX_FACES,

    UNKNOWN_FACES_DIR,
    UNKNOWN_MIN_TRACK_FRAMES,
    UNKNOWN_MAX_SAMPLES,
    UNKNOWN_CAPTURE_INTERVAL,
    UNKNOWN_MIN_QUALITY,
)

from database import (
    add_unknown_sample,
    update_unknown_image,
    log_audio,
    log_recognition,
)

from face_core import (
    FaceIndex,
    extract_embedding,
    face_quality,
    register_unknown,
    find_matching_unknown,
)

from tracking import (
    CentroidTracker,
)

from video_processor import (
    bbox_from_landmarks,
    lip_open_ratio,
)


# ============================================================
# Recognition stability
# ============================================================

MAX_RECOGNITION_FAILURES = 3


# ============================================================
# Speaking state
# ============================================================

class SpeakingState:

    def __init__(self):
        self.value = False

    def set(self, value):
        self.value = value


# ============================================================
# MediaPipe Face Landmarker
# ============================================================

def create_face_landmarker():

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

    options = (
        mp.tasks.vision.FaceLandmarkerOptions(
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
    )

    return (
        mp.tasks.vision.FaceLandmarker
        .create_from_options(options)
    )


# ============================================================
# Prepare recognition crop
# ============================================================

def prepare_recognition_crop(crop):

    if crop is None:
        return None

    if crop.size == 0:
        return None

    face_height, face_width = (
        crop.shape[:2]
    )

    # --------------------------------------------------------
    # Too small for reliable ArcFace recognition.
    # --------------------------------------------------------

    if (
        face_width
        < MIN_RECOGNITION_FACE_SIZE
        or
        face_height
        < MIN_RECOGNITION_FACE_SIZE
    ):
        return None

    recognition_crop = crop

    # --------------------------------------------------------
    # Upscale small but usable face crops.
    # --------------------------------------------------------

    if (
        face_width
        < FACE_UPSCALE_THRESHOLD
        or
        face_height
        < FACE_UPSCALE_THRESHOLD
    ):

        recognition_crop = cv2.resize(
            crop,
            None,
            fx=FACE_UPSCALE_FACTOR,
            fy=FACE_UPSCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC,
        )

    return recognition_crop


# ============================================================
# Realtime pipeline
# ============================================================

def run(
    camera=0,
    width=1280,
    height=720,
):

    logging.info(
        "Starting realtime recognition."
    )

    # ========================================================
    # Open camera
    # ========================================================

    cap = cv2.VideoCapture(
        camera,
        cv2.CAP_MSMF,
    )

    if not cap.isOpened():

        logging.warning(
            "Could not open camera %s using MSMF. "
            "Trying DirectShow...",
            camera,
        )

        cap.release()

        cap = cv2.VideoCapture(
            camera,
            cv2.CAP_DSHOW,
        )

    if not cap.isOpened():

        logging.warning(
            "Could not open camera %s using DirectShow. "
            "Trying default OpenCV backend...",
            camera,
        )

        cap.release()

        cap = cv2.VideoCapture(
            camera
        )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open camera {camera}. "
            "Please check camera permissions, "
            "camera index, and whether another "
            "application is using the camera."
        )

    # ========================================================
    # Camera resolution
    # ========================================================

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        width,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        height,
    )

    actual_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if actual_fps <= 0:
        actual_fps = 30.0

    logging.info(
        "Camera opened: %sx%s @ %.2f FPS",
        actual_width,
        actual_height,
        actual_fps,
    )

    # ========================================================
    # Face index
    # ========================================================

    index = FaceIndex()

    tracker = CentroidTracker()

    # ========================================================
    # Realtime VAD
    # ========================================================

    audio_state = (
        SpeakingState()
    )

    vad = RealtimeVAD(
        callback=audio_state.set
    )

    vad.start()

    # ========================================================
    # MediaPipe
    # ========================================================

    mesh = create_face_landmarker()

    # ========================================================
    # Runtime state
    # ========================================================

    frame_no = 0

    last_speech_start = None

    last_speaker_id = None

    last_recognition_log = {}

    try:

        while True:

            # =================================================
            # Read camera frame
            # =================================================

            ok, frame = cap.read()

            if not ok:
                break

            frame_no += 1

            timestamp = time.monotonic()

            # =================================================
            # BGR → RGB
            # =================================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            # =================================================
            # MediaPipe image
            # =================================================

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb,
            )

            timestamp_ms = int(
                frame_no
                * 1000
                / actual_fps
            )

            result = (
                mesh.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            )

            detections = []

            # =================================================
            # Face detections
            # =================================================

            for landmarks in (
                result.face_landmarks
            ):

                if not landmarks:
                    continue

                bbox = bbox_from_landmarks(
                    landmarks,
                    frame.shape,
                )

                x0, y0, x1, y1 = bbox

                face_width = (
                    x1 - x0
                )

                face_height = (
                    y1 - y0
                )

                # ------------------------------------------------
                # Detection/tracking minimum.
                # ------------------------------------------------

                if (
                    face_width
                    < MIN_FACE_SIZE
                    or
                    face_height
                    < MIN_FACE_SIZE
                ):
                    continue

                detections.append(
                    {
                        "bbox": bbox,

                        "landmarks": landmarks,

                        "lip_open":
                            lip_open_ratio(
                                landmarks
                            ),
                    }
                )

            # =================================================
            # Tracking
            # =================================================

            assignments = tracker.update(
                detections,
                frame_no,
            )

            candidates = []

            # =================================================
            # Process tracked faces
            # =================================================

            for detection, track in assignments:

                # ------------------------------------------------
                # Initialize state
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
                    "unknown_samples",
                ):

                    track.unknown_samples = 0

                if not hasattr(
                    track,
                    "last_unknown_capture",
                ):

                    track.last_unknown_capture = (
                        -10_000
                    )

                if not hasattr(
                    track,
                    "registered_unknown",
                ):

                    track.registered_unknown = (
                        False
                    )

                if not hasattr(
                    track,
                    "unknown_id",
                ):

                    track.unknown_id = None

                # =================================================
                # Bounding box / crop
                # =================================================

                x0, y0, x1, y1 = (
                    detection["bbox"]
                )

                face_width = (
                    x1 - x0
                )

                face_height = (
                    y1 - y0
                )

                crop = frame[
                    y0:y1,
                    x0:x1,
                ]

                if crop.size == 0:
                    continue

                # =================================================
                # Determine whether recognition is possible
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

                # =================================================
                # Face recognition
                # =================================================

                if (
                    can_recognize
                    and should_recognize
                ):

                    recognition_crop = (
                        prepare_recognition_crop(
                            crop
                        )
                    )

                    if (
                        recognition_crop
                        is not None
                    ):

                        embedding = (
                            extract_embedding(
                                recognition_crop
                            )
                        )

                        if (
                            embedding
                            is not None
                        ):

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
                                # Previously recognized person:
                                # keep identity temporarily.
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
                                # mark Unknown.
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
                # Unknown face sampling
                # =================================================

                if (
                    track.person_id is None
                    and
                    track.person_name == "Unknown"
                ):

                    # ------------------------------------------------
                    # Only count actual recognition attempts.
                    # Very tiny faces should not be repeatedly
                    # classified as unknown.
                    # ------------------------------------------------

                    if can_recognize:

                        track.unknown_confirmations += 1

                    if (
                        track.unknown_confirmations
                        >= UNKNOWN_MIN_TRACK_FRAMES

                        and
                        track.unknown_samples
                        < UNKNOWN_MAX_SAMPLES

                        and
                        frame_no
                        - track.last_unknown_capture
                        >= UNKNOWN_CAPTURE_INTERVAL
                    ):

                        quality = face_quality(
                            crop
                        )

                        if (
                            quality
                            >= UNKNOWN_MIN_QUALITY
                        ):

                            unknown_dir = (
                                UNKNOWN_FACES_DIR
                            )

                            unknown_dir.mkdir(
                                parents=True,
                                exist_ok=True,
                            )

                            sample_number = (
                                track.unknown_samples
                                + 1
                            )

                            image_path = (
                                unknown_dir
                                /
                                (
                                    f"track_"
                                    f"{track.track_id}"
                                    f"_sample_"
                                    f"{sample_number}"
                                    f".jpg"
                                )
                            )

                            embedding_path = (
                                unknown_dir
                                /
                                (
                                    f"track_"
                                    f"{track.track_id}"
                                    f"_sample_"
                                    f"{sample_number}"
                                    f".npy"
                                )
                            )

                            # ------------------------------------------------
                            # Save image
                            # ------------------------------------------------

                            success = cv2.imwrite(
                                str(image_path),
                                crop,
                            )

                            if not success:

                                logging.warning(
                                    "Could not save "
                                    "unknown face image: %s",
                                    image_path,
                                )

                                continue

                            # ------------------------------------------------
                            # Save embedding
                            # ------------------------------------------------

                            np.save(
                                embedding_path,
                                track.embedding,
                            )

                            # ------------------------------------------------
                            # First sample:
                            # create/find unknown identity.
                            # ------------------------------------------------

                            if (
                                not
                                track.registered_unknown
                            ):

                                existing_unknown = (
                                    find_matching_unknown(
                                        track.embedding
                                    )
                                )

                                if (
                                    existing_unknown
                                    is not None
                                ):

                                    track.unknown_id = (
                                        existing_unknown[
                                            "unknown_id"
                                        ]
                                    )

                                    track.registered_unknown = (
                                        True
                                    )

                                    logging.info(
                                        "Unknown face "
                                        "matched existing "
                                        "identity: "
                                        "unknown_id=%s "
                                        "similarity=%.3f",
                                        track.unknown_id,
                                        existing_unknown[
                                            "similarity"
                                        ],
                                    )

                                else:

                                    label = (
                                        f"track_"
                                        f"{track.track_id}"
                                    )

                                    unknown_id = (
                                        register_unknown(
                                            label=label,
                                            embedding=(
                                                track.embedding
                                            ),
                                            image_path=(
                                                image_path
                                            ),
                                        )
                                    )

                                    track.unknown_id = (
                                        unknown_id
                                    )

                                    track.registered_unknown = (
                                        True
                                    )

                                    logging.info(
                                        "Created new "
                                        "unknown identity: "
                                        "unknown_id=%s "
                                        "label=%s",
                                        unknown_id,
                                        label,
                                    )

                            # ------------------------------------------------
                            # Update database
                            # ------------------------------------------------

                            if (
                                track.unknown_id
                                is not None
                            ):

                                update_unknown_image(
                                    unknown_id=(
                                        track.unknown_id
                                    ),
                                    image_path=(
                                        image_path
                                    ),
                                )

                                add_unknown_sample(
                                    unknown_id=(
                                        track.unknown_id
                                    ),
                                    image_path=(
                                        image_path
                                    ),
                                    embedding_path=(
                                        embedding_path
                                    ),
                                    quality=quality,
                                )

                                track.unknown_samples += 1

                                track.last_unknown_capture = (
                                    frame_no
                                )

                # =================================================
                # Lip movement
                # =================================================

                track.lip_open = (
                    detection[
                        "lip_open"
                    ]
                )

                # =================================================
                # Active speaker candidate
                # =================================================

                if (
                    audio_state.value
                    and
                    track.lip_open
                    >= LIP_OPEN_THRESHOLD
                ):

                    candidates.append(
                        track
                    )

                # =================================================
                # Recognition logging
                # =================================================

                last_time = (
                    last_recognition_log.get(
                        track.track_id,
                        0,
                    )
                )

                if (
                    timestamp - last_time
                    >= 1.0
                ):

                    log_recognition(
                        timestamp_sec=timestamp,
                        person_id=track.person_id,
                        person_name=track.person_name,
                        confidence=track.confidence,
                        source="realtime",
                    )

                    last_recognition_log[
                        track.track_id
                    ] = timestamp

                # =================================================
                # Display
                # =================================================

                if (
                    track.person_name
                    == "Unknown"
                ):

                    color = (
                        0,
                        0,
                        255,
                    )

                elif (
                    track.person_id
                    is None
                ):

                    color = (
                        0,
                        165,
                        255,
                    )

                else:

                    color = (
                        0,
                        255,
                        0,
                    )

                cv2.rectangle(
                    frame,
                    (x0, y0),
                    (x1, y1),
                    color,
                    2,
                )

                # ------------------------------------------------
                # Label
                # ------------------------------------------------

                if (
                    not can_recognize
                    and
                    track.person_id
                    is None
                ):

                    label = (
                        "Recognition Pending"
                    )

                else:

                    label = (
                        track.person_name
                        or
                        "Unknown"
                    )

                    if (
                        track.person_id
                        is not None
                    ):

                        label += (
                            f" "
                            f"{track.confidence:.2f}"
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
            # Select active speaker
            # =====================================================

            if (
                audio_state.value
                and
                candidates
            ):

                speaker = max(
                    candidates,
                    key=lambda t:
                        t.lip_open,
                )

                if (
                    speaker.person_id
                    is not None
                ):

                    if (
                        last_speaker_id
                        != speaker.person_id
                    ):

                        last_speaker_id = (
                            speaker.person_id
                        )

                        last_speech_start = (
                            time.time()
                        )

                    cv2.putText(
                        frame,
                        (
                            f"SPEAKING: "
                            f"{speaker.person_name}"
                        ),
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

                    cv2.rectangle(
                        frame,
                        (
                            int(
                                speaker.center_x
                                - 80
                            ),
                            int(
                                speaker.center_y
                                - 100
                            ),
                        ),
                        (
                            int(
                                speaker.center_x
                                + 80
                            ),
                            int(
                                speaker.center_y
                                + 100
                            ),
                        ),
                        (0, 255, 0),
                        3,
                    )

            else:

                if (
                    last_speaker_id
                    is not None
                    and
                    last_speech_start
                    is not None
                ):

                    speaker_name = next(
                        (
                            t.person_name
                            for t
                            in tracker.tracks.values()
                            if (
                                t.person_id
                                == last_speaker_id
                            )
                        ),
                        "Unknown",
                    )

                    speaker_confidence = next(
                        (
                            t.confidence
                            for t
                            in tracker.tracks.values()
                            if (
                                t.person_id
                                == last_speaker_id
                            )
                        ),
                        0.0,
                    )

                    log_audio(
                        last_speech_start,
                        time.time(),
                        last_speaker_id,
                        speaker_name,
                        speaker_confidence,
                        "realtime",
                    )

                last_speaker_id = None

                last_speech_start = None

            # =================================================
            # Microphone status
            # =================================================

            cv2.putText(
                frame,
                (
                    "MIC: SPEECH"
                    if audio_state.value
                    else
                    "MIC: SILENT"
                ),
                (
                    20,
                    frame.shape[0] - 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    (0, 255, 0)
                    if audio_state.value
                    else
                    (255, 255, 255)
                ),
                2,
            )

            # =================================================
            # Display
            # =================================================

            cv2.imshow(
                "AI Face + Active Speaker Recognition",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if (
                key == ord("q")
                or key == 27
            ):
                break

    finally:

        # =====================================================
        # Save final audio event
        # =====================================================

        if (
            last_speaker_id
            is not None
            and
            last_speech_start
            is not None
        ):

            speaker_name = next(
                (
                    t.person_name
                    for t
                    in tracker.tracks.values()
                    if (
                        t.person_id
                        == last_speaker_id
                    )
                ),
                "Unknown",
            )

            speaker_confidence = next(
                (
                    t.confidence
                    for t
                    in tracker.tracks.values()
                    if (
                        t.person_id
                        == last_speaker_id
                    )
                ),
                0.0,
            )

            log_audio(
                last_speech_start,
                time.time(),
                last_speaker_id,
                speaker_name,
                speaker_confidence,
                "realtime",
            )

        # =====================================================
        # Cleanup
        # =====================================================

        vad.stop()

        cap.release()

        mesh.close()

        cv2.destroyAllWindows()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    run(
        camera=args.camera
    )