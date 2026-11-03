"""Small structured logging helpers shared by the training pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time


def log_event(event: str, *, level: str = "INFO", **fields) -> dict:
    """Write one timestamped JSON event to stdout and flush it immediately."""
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "level": level.upper(),
        "event": event,
        **fields,
    }
    print(json.dumps(record, sort_keys=True, ensure_ascii=True, default=str), flush=True)
    return record


def replace_with_retry(source: str | Path, destination: str | Path, *, attempts: int = 7) -> None:
    """Retry transient Windows sharing/permission denials during atomic publish."""
    source, destination = Path(source), Path(destination)
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.05 * (2 ** attempt), 0.5))


def write_json_atomic(path: str | Path, payload: dict) -> None:
    """Publish a complete JSON status file so readers never see a partial write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                          dir=path.parent)
    tmp = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
            stream.flush()
        replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
