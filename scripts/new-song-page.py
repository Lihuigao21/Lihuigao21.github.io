#!/usr/bin/env python3
"""Render the fixed song display template as standalone HTML; no site build step."""
import argparse
from datetime import date
from html import escape
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "scripts/templates/song-display.html"


def text(config, key):
    value = config[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be nonempty text")
    return value


def paragraphs(items):
    if not isinstance(items, list) or not items or any(not isinstance(p, str) for p in items):
        raise ValueError("Paragraphs must be a nonempty list of text strings")
    return "\n".join("<p>" + escape(p).replace("\n", "<br>") + "</p>" for p in items)


def asset_url(value, prefix):
    url = urlsplit(value)
    if (url.scheme or url.netloc or url.fragment or "\\" in value or
            not re.fullmatch(r"assets/(media|data)/[A-Za-z0-9_./-]+", url.path)):
        raise ValueError("Use a repository-relative assets/media or assets/data URL")
    path = (ROOT / url.path).resolve()
    if not path.is_relative_to(ROOT / "assets") or not path.is_file():
        raise ValueError("Missing local asset: " + url.path)
    return escape(prefix + "/" + value)


def render(config, output):
    title = text(config, "song")
    performer = text(config, "performer")
    published = date.fromisoformat(text(config, "date"))
    duration = config["duration_seconds"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_seconds must be a positive finite number")
    mime = config["media_type"]
    if mime not in {"audio/mp4", "audio/mpeg", "audio/ogg", "audio/wav", "audio/webm", "audio/flac"}:
        raise ValueError("media_type must describe an audio file; inspect MP4 tracks first")
    lyrics = config["lyrics"]
    note = config["note"]
    lyric_html = paragraphs(lyrics)
    note_html = paragraphs(note)
    prefix = Path(os.path.relpath(ROOT, output.parent)).as_posix()
    timing_attr = ""
    timing_description = ""
    if config.get("timings"):
        timing_url = asset_url(config["timings"], prefix)
        timing_path = ROOT / urlsplit(config["timings"]).path
        starts = json.loads(timing_path.read_text(encoding="utf-8-sig"))["lineStarts"]
        if (not isinstance(starts, list) or len(starts) != sum(len(p.split("\n")) for p in lyrics) or
                any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 or t >= duration for t in starts) or
                starts != sorted(starts)):
            raise ValueError("lineStarts must match every lyric line, be ordered, and precede the media end")
        timing_attr = f' data-timings="{timing_url}"'
        description = text(config, "timing_note")
        timing_description = "<p>" + escape(description)
        if config.get("timing_source_url"):
            source = config["timing_source_url"]
            parsed = urlsplit(source)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
                raise ValueError("timing_source_url must be a public HTTP(S) URL")
            timing_description += f' <a href="{escape(source)}">时间轴来源</a>。'
        timing_description += "</p>"
    values = {
        "ROOT": prefix,
        "ART_INDEX": Path(os.path.relpath(ROOT / "life/art-reviews.html", output.parent)).as_posix(),
        "TITLE": escape(title), "PERFORMER": escape(performer),
        "RECORD_TITLE": escape(title.upper()), "RECORD_PERFORMER": escape(performer.upper()),
        "DATE": published.isoformat(), "DISPLAY_DATE": published.strftime("%Y.%m.%d"),
        "DURATION": str(duration), "DURATION_TEXT": f"{int(duration) // 60}:{int(duration) % 60:02}",
        "MEDIA": asset_url(text(config, "media"), prefix), "MEDIA_TYPE": escape(mime),
        "LYRICS_LANG": escape(config.get("lyrics_language", "zh-CN")),
        "LYRICS": "<p>" + escape(title) + "</p>\n" + lyric_html,
        "NOTE": note_html, "CREDIT": escape(text(config, "credit")),
        "TIMINGS_ATTR": timing_attr, "TIMING_DESCRIPTION": timing_description,
    }
    template = TEMPLATE.read_text(encoding="utf-8")
    return re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: values[m[1]], template)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Private UTF-8 JSON content file")
    parser.add_argument("--output", required=True, type=Path, help="New HTML file inside this repository")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if not output.is_relative_to(ROOT) or output.suffix != ".html":
            raise ValueError("Output must be an HTML file inside this repository")
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        html = render(config, output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(html)
        print("Created " + output.relative_to(ROOT).as_posix())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
