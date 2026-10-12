"""Optional, validated readable-transcript refinement via local Ollama."""

from dataclasses import dataclass
import json
import logging
import re
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import config


logger = logging.getLogger(__name__)


class TranscriptRefinementError(RuntimeError):
    """Raised when the configured local transcript refiner cannot be used."""


@dataclass(frozen=True)
class RefinedGroup:
    source_segment_ids: tuple[int, ...]
    text: str


@dataclass(frozen=True)
class SpeakerChangeRecommendation:
    after_segment_id: int
    reason: str


@dataclass(frozen=True)
class RefinementResult:
    groups: tuple[RefinedGroup, ...]
    speaker_change_recommendations: tuple[SpeakerChangeRecommendation, ...]


_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["groups", "speaker_change_recommendations"],
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source_segment_ids", "text"],
                "properties": {
                    "source_segment_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "text": {"type": "string"},
                },
            },
        },
        "speaker_change_recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["after_segment_id", "reason"],
                "properties": {
                    "after_segment_id": {"type": "integer"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}


def _chunk_ranges(segments: Sequence[dict]) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    while start < len(segments):
        end = start
        char_count = 0
        while end < len(segments) and end - start < config.TRANSCRIPT_REFINEMENT_CHUNK_SEGMENTS:
            next_size = len(str(segments[end]["text"]))
            if end > start and char_count + next_size > config.TRANSCRIPT_REFINEMENT_CHUNK_CHARS:
                break
            char_count += next_size
            end += 1
        if end == start:
            end += 1
        ranges.append((start, end))
        start = end
    return ranges


def _post_ollama_chat(user_content: str) -> dict:
    payload = {
        "model": config.TRANSCRIPT_REFINEMENT_MODEL,
        "stream": False,
        "format": _OUTPUT_SCHEMA,
        "options": {"temperature": 0},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You format an existing transcript in its original language. "
                    "Do not translate, add, omit, or infer spoken content. Correct "
                    "only punctuation, capitalization, spacing, and obvious text "
                    "formatting. Group source segments only when they are consecutive "
                    "and the input speaker labels are identical. Never change a "
                    "speaker label or infer identity from words. Output each target "
                    "segment ID exactly once. Context segments are read-only. "
                    "A speaker-change recommendation is only a request for human "
                    "audio review and must never assert a person's identity."
                ),
            },
            {"role": "user", "content": user_content},
        ],
    }
    request = Request(
        f"{config.TRANSCRIPT_REFINEMENT_URL}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS) as response:
            envelope = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TranscriptRefinementError(
            f"Local Ollama request failed ({config.TRANSCRIPT_REFINEMENT_URL}): {error}"
        ) from error

    try:
        content = envelope["message"]["content"]
        result = json.loads(content)
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise TranscriptRefinementError(
            "Local Ollama returned an invalid structured response"
        ) from error
    if not isinstance(result, dict):
        raise TranscriptRefinementError("Refinement response must be a JSON object")
    return result


def _normalize_and_validate_response(
    response: dict,
    targets: Sequence[dict],
    all_segments: Sequence[dict],
) -> tuple[list[RefinedGroup], list[SpeakerChangeRecommendation]]:
    target_ids = [int(item["segment_id"]) for item in targets]
    target_id_set = set(target_ids)
    positions = {
        int(segment["segment_id"]): index
        for index, segment in enumerate(all_segments)
    }
    by_id = {int(segment["segment_id"]): segment for segment in all_segments}
    groups_payload = response.get("groups")
    recommendations_payload = response.get("speaker_change_recommendations")
    if not isinstance(groups_payload, list) or not isinstance(recommendations_payload, list):
        raise TranscriptRefinementError("Refinement response is missing required arrays")

    groups = []
    seen_ids = []
    previous_group_position = -1
    for payload in groups_payload:
        if not isinstance(payload, dict):
            raise TranscriptRefinementError("Each refined group must be an object")
        refs = payload.get("source_segment_ids")
        text = payload.get("text")
        if (
            not isinstance(refs, list)
            or not refs
            or any(isinstance(ref, bool) or not isinstance(ref, int) for ref in refs)
            or len(set(refs)) != len(refs)
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise TranscriptRefinementError("A refined group has invalid references or text")
        ids = tuple(refs)
        if any(segment_id not in target_id_set for segment_id in ids):
            raise TranscriptRefinementError("Refinement referenced a non-target segment")
        id_positions = [positions[segment_id] for segment_id in ids]
        if id_positions != list(range(id_positions[0], id_positions[0] + len(ids))):
            raise TranscriptRefinementError("Refinement may only group consecutive segments")
        if id_positions[0] <= previous_group_position:
            raise TranscriptRefinementError("Refined groups are out of chronological order")
        previous_group_position = id_positions[-1]
        first = by_id[ids[0]]
        previous = first
        for segment_id in ids[1:]:
            current = by_id[segment_id]
            gap = int(current["start_ms"]) - int(previous["end_ms"])
            if (
                current["speaker_label"] != first["speaker_label"]
                or gap < 0
                or gap > config.TRANSCRIPT_MERGE_MAX_GAP_MS
            ):
                raise TranscriptRefinementError(
                    "Refinement attempted to merge across a speaker change, "
                    "overlap, or long pause"
                )
            previous = current
        input_text = " ".join(by_id[segment_id]["text"].strip() for segment_id in ids)
        if len(text.strip()) > len(input_text) * 1.5 + 64:
            raise TranscriptRefinementError("Refined text is too different in length from its source")
        source_words = re.findall(r"[^\W_]+(?:['’][^\W_]+)*", input_text.casefold())
        refined_words = re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text.casefold())
        if source_words != refined_words:
            raise TranscriptRefinementError(
                "Refinement changed spoken words; only formatting changes are allowed"
            )
        groups.append(RefinedGroup(ids, text.strip()))
        seen_ids.extend(ids)

    if seen_ids != target_ids:
        raise TranscriptRefinementError(
            "Refinement omitted, duplicated, or reordered source segment references"
        )

    recommendations = []
    for recommendation in recommendations_payload:
        if not isinstance(recommendation, dict):
            raise TranscriptRefinementError("Speaker review recommendation must be an object")
        after_segment_id = recommendation.get("after_segment_id")
        reason = recommendation.get("reason")
        if (
            isinstance(after_segment_id, bool)
            or not isinstance(after_segment_id, int)
            or after_segment_id not in target_id_set
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise TranscriptRefinementError("Speaker review recommendation is invalid")
        recommendations.append(
            SpeakerChangeRecommendation(after_segment_id, reason.strip()[:300])
        )
    return groups, recommendations


def refine_transcript(segments: Sequence[dict]) -> RefinementResult:
    """Format transcript text while retaining strict source/speaker boundaries."""
    if not segments:
        return RefinementResult((), ())
    normalized = []
    seen = set()
    for segment in segments:
        segment_id = segment.get("segment_id")
        start_ms, end_ms = segment.get("start_ms"), segment.get("end_ms")
        text = segment.get("text")
        if (
            isinstance(segment_id, bool)
            or not isinstance(segment_id, int)
            or segment_id in seen
            or isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or start_ms < 0
            or end_ms <= start_ms
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise TranscriptRefinementError("Detailed transcript contains invalid source data")
        seen.add(segment_id)
        normalized.append({
            **segment,
            "segment_id": segment_id,
            "speaker_label": str(segment.get("speaker_label") or "UNKNOWN"),
            "text": text.strip(),
        })
    normalized.sort(key=lambda item: (item["start_ms"], item["end_ms"], item["segment_id"]))
    if config.TRANSCRIPT_REFINEMENT_CHUNK_SEGMENTS < 1:
        raise TranscriptRefinementError("Chunk segment limit must be positive")

    output_groups = []
    output_recommendations = []
    ranges = _chunk_ranges(normalized)
    for chunk_index, (start, end) in enumerate(ranges):
        targets = normalized[start:end]
        context = normalized[max(0, start - 2):start] + normalized[end:end + 2]
        response = _post_ollama_chat(json.dumps(
            {
                "target_segments": targets,
                "read_only_context_segments": context,
                "target_segment_ids": [item["segment_id"] for item in targets],
            },
            ensure_ascii=False,
        ))
        groups, recommendations = _normalize_and_validate_response(
            response,
            targets,
            normalized,
        )
        output_groups.extend(groups)
        output_recommendations.extend(recommendations)
        logger.debug(
            "Refined transcript chunk %d/%d (%d segments)",
            chunk_index + 1,
            len(ranges),
            len(targets),
        )
    return RefinementResult(tuple(output_groups), tuple(output_recommendations))
