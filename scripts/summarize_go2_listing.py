#!/usr/bin/env python3
"""Summarize a Go2 Spark /logs listing exported by _tmp_spark_current_listing."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


LISTING_DATE_RE = re.compile(r"^date_utc=(?P<date>.+)$")
RUN_DIR_RE = re.compile(r"^/logs/rsl_rl/(?P<experiment>[^/]+)/(?P<run>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")
FILE_RE = re.compile(
    r"^\S+\s+\S+\s+\S+\s+\S+\s+(?P<size>\d+)\s+"
    r"(?P<month>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<hour>\d{2}):(?P<minute>\d{2})\s+"
    r"(?P<path>/logs/rsl_rl/[^ ]+)$"
)
MODEL_RE = re.compile(r"model_(?P<iteration>\d+)\.pt$")
VIDEO_RE = re.compile(r"rl-video-step-(?P<step>\d+)\.mp4$")
LEARNING_ITERATION_RE = re.compile(r"Learning iteration (?P<iteration>\d+)/(?P<total>\d+)")
TOTAL_TIMESTEPS_RE = re.compile(r"Total timesteps:\s+(?P<steps>\d+)")
COMPUTATION_RE = re.compile(r"Computation:\s+(?P<sps>\d+)\s+steps/s")
HEARTBEAT_RE = re.compile(
    r"event=heartbeat .*elapsed_seconds=(?P<elapsed>[0-9.]+) .*progress=(?P<progress>\w+)"
)

MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


@dataclass(frozen=True)
class Artifact:
    path: str
    size: int
    timestamp: datetime
    value: int


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("listing", type=Path, help="Path to logs-listing.txt")
    parser.add_argument("--required-hours", type=float, default=12.0)
    parser.add_argument("--max-video-gap-minutes", type=float, default=31.5)
    parser.add_argument("--max-checkpoint-gap-minutes", type=float, default=31.5)
    return parser.parse_args(argv)


def parse_listing_date(lines: list[str]) -> datetime:
    for line in lines:
        match = LISTING_DATE_RE.match(line)
        if match:
            return datetime.fromisoformat(match.group("date").replace("Z", "+00:00"))
    raise ValueError("listing does not contain date_utc")


def parse_active_run(lines: list[str]) -> str:
    for index, line in enumerate(lines):
        if line.strip() == "== newest current run files ==":
            for candidate in lines[index + 1 :]:
                candidate = candidate.strip()
                if not candidate:
                    continue
                if RUN_DIR_RE.match(candidate):
                    return candidate
                if candidate.startswith("== "):
                    break
    raise ValueError("listing does not contain an active run after 'newest current run files'")


def run_started_at(run_dir: str) -> datetime:
    match = RUN_DIR_RE.match(run_dir)
    if not match:
        raise ValueError(f"cannot parse run directory timestamp: {run_dir}")
    return datetime.strptime(match.group("run"), "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc)


def parse_file_timestamp(match: re.Match[str], listing_date: datetime) -> datetime:
    return datetime(
        listing_date.year,
        MONTHS[match.group("month")],
        int(match.group("day")),
        int(match.group("hour")),
        int(match.group("minute")),
        tzinfo=timezone.utc,
    )


def parse_artifacts(lines: list[str], active_run: str, listing_date: datetime) -> tuple[list[Artifact], list[Artifact]]:
    checkpoints: list[Artifact] = []
    videos: list[Artifact] = []
    for line in lines:
        match = FILE_RE.match(line)
        if not match:
            continue
        path = match.group("path")
        if not path.startswith(active_run + "/"):
            continue
        timestamp = parse_file_timestamp(match, listing_date)
        size = int(match.group("size"))
        name = Path(path).name
        model_match = MODEL_RE.match(name)
        if model_match:
            checkpoints.append(Artifact(path, size, timestamp, int(model_match.group("iteration"))))
            continue
        video_match = VIDEO_RE.match(name)
        if video_match:
            videos.append(Artifact(path, size, timestamp, int(video_match.group("step"))))
    checkpoints.sort(key=lambda item: (item.value, item.timestamp))
    videos.sort(key=lambda item: (item.value, item.timestamp))
    return checkpoints, videos


def latest_metric(lines: list[str], regex: re.Pattern[str], group: str) -> str | None:
    value = None
    for line in lines:
        match = regex.search(line)
        if match:
            value = match.group(group)
    return value


def max_gap_minutes(items: list[Artifact]) -> float | None:
    if len(items) < 2:
        return None
    gaps = [
        (current.timestamp - previous.timestamp).total_seconds() / 60.0
        for previous, current in zip(items, items[1:])
    ]
    return max(gaps)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    lines = args.listing.read_text(encoding="utf-8", errors="replace").splitlines()
    listing_date = parse_listing_date(lines)
    active_run = parse_active_run(lines)
    start_time = run_started_at(active_run)
    age_hours = (listing_date - start_time).total_seconds() / 3600.0
    checkpoints, videos = parse_artifacts(lines, active_run, listing_date)

    latest_checkpoint = checkpoints[-1] if checkpoints else None
    latest_video = videos[-1] if videos else None
    checkpoint_gap = max_gap_minutes(checkpoints)
    video_gap = max_gap_minutes(videos)
    learning_iteration = latest_metric(lines, LEARNING_ITERATION_RE, "iteration")
    total_timesteps = latest_metric(lines, TOTAL_TIMESTEPS_RE, "steps")
    sps = latest_metric(lines, COMPUTATION_RE, "sps")

    latest_heartbeat = None
    for line in lines:
        match = HEARTBEAT_RE.search(line)
        if match:
            latest_heartbeat = match.groupdict()

    checks: list[tuple[str, bool, str]] = [
        (
            "run_age",
            age_hours >= args.required_hours,
            f"{age_hours:.2f}h / required {args.required_hours:.2f}h",
        ),
        ("checkpoint_exists", latest_checkpoint is not None, latest_checkpoint.path if latest_checkpoint else "missing"),
        ("video_exists", latest_video is not None, latest_video.path if latest_video else "missing"),
    ]
    if checkpoint_gap is not None:
        checks.append(
            (
                "checkpoint_gap",
                checkpoint_gap <= args.max_checkpoint_gap_minutes,
                f"max {checkpoint_gap:.1f}m / allowed {args.max_checkpoint_gap_minutes:.1f}m",
            )
        )
    else:
        checks.append(("checkpoint_gap", False, "need at least two checkpoints"))
    if video_gap is not None:
        checks.append(
            (
                "video_gap",
                video_gap <= args.max_video_gap_minutes,
                f"max {video_gap:.1f}m / allowed {args.max_video_gap_minutes:.1f}m",
            )
        )
    else:
        checks.append(("video_gap", False, "need at least two videos"))
    if latest_heartbeat:
        checks.append(("artifact_progress", latest_heartbeat["progress"] == "advanced", str(latest_heartbeat)))
    else:
        checks.append(("artifact_progress", False, "missing heartbeat"))

    print(f"listing_date={listing_date.isoformat()}")
    print(f"active_run={active_run}")
    print(f"run_age_hours={age_hours:.2f}")
    if learning_iteration:
        print(f"latest_learning_iteration={learning_iteration}")
    if total_timesteps:
        print(f"latest_total_timesteps={total_timesteps}")
    if sps:
        print(f"latest_sps={sps}")
    if latest_checkpoint:
        print(f"latest_checkpoint=model_{latest_checkpoint.value}.pt time={latest_checkpoint.timestamp.isoformat()}")
    if latest_video:
        print(f"latest_video=rl-video-step-{latest_video.value}.mp4 time={latest_video.timestamp.isoformat()}")
    if checkpoint_gap is not None:
        print(f"max_checkpoint_gap_minutes={checkpoint_gap:.1f}")
    if video_gap is not None:
        print(f"max_video_gap_minutes={video_gap:.1f}")
    for name, passed, detail in checks:
        print(f"check.{name}={'PASS' if passed else 'FAIL'} {detail}")

    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
