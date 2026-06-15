import json
import re
from html import unescape

import requests
import yt_dlp

LANGUAGE_PREFERENCES = ("en", "en-US", "en-GB")


def _strip_html(text: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _pick_caption_track(info: dict, preferred_languages: tuple[str, ...] = LANGUAGE_PREFERENCES) -> dict | None:
    for group_name in ("subtitles", "automatic_captions"):
        group = info.get(group_name) or {}
        languages = list(preferred_languages) + [lang for lang in group if lang not in preferred_languages]
        for language in languages:
            tracks = group.get(language) or []
            json_track = next((track for track in tracks if track.get("ext") == "json3"), None)
            vtt_track = next((track for track in tracks if track.get("ext") == "vtt"), None)
            track = json_track or vtt_track
            if track and track.get("url"):
                return track
    return None


def _parse_json3(text: str) -> str:
    payload = json.loads(text)
    parts = []
    for event in payload.get("events", []):
        segment_text = "".join(seg.get("utf8", "") for seg in event.get("segs", []))
        segment_text = segment_text.replace("\n", " ").strip()
        if segment_text:
            parts.append(segment_text)
    return " ".join(parts)


def _parse_vtt(text: str) -> str:
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or "-->" in line or line.isdigit():
            continue
        line = _strip_html(line)
        if line:
            parts.append(line)
    return " ".join(parts)


def fetch_youtube_transcript(url: str) -> str | None:
    ydl_opts = {"quiet": True, "skip_download": True, "writesubtitles": True, "writeautomaticsub": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    track = _pick_caption_track(info)
    if not track:
        return None

    response = requests.get(track["url"], timeout=60)
    response.raise_for_status()

    if track.get("ext") == "json3":
        transcript = _parse_json3(response.text)
    else:
        transcript = _parse_vtt(response.text)

    transcript = " ".join(transcript.split())
    return transcript or None
