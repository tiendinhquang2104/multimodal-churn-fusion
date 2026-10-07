"""Verified copy of a completed temporal SSL run to persistent storage."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


REQUIRED_FILES = (
    "features_temporal_ssl.parquet",
    "features_temporal_ssl_metadata.parquet",
    "manifest_temporal_ssl.json",
    "checkpoints/best_transaction_ssl.pt",
    "checkpoints/best_text_ssl.pt",
    "checkpoints/best_image_ssl.pt",
)
COMPLETION_MARKER = "_drive_copy_complete.json"


def copy_verified_file(source: Path, destination: Path) -> tuple[int, str]:
    """Stream a file, then verify the bytes on the destination filesystem."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different files")
    temporary = destination.with_name(destination.name + ".partial")
    source_hash = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            while chunk := reader.read(8 * 1024 * 1024):
                writer.write(chunk)
                source_hash.update(chunk)
                size += len(chunk)
            writer.flush()
        copied_hash = hashlib.sha256()
        with temporary.open("rb") as reader:
            while chunk := reader.read(8 * 1024 * 1024):
                copied_hash.update(chunk)
        if temporary.stat().st_size != size or copied_hash.digest() != source_hash.digest():
            raise IOError(f"Drive copy verification failed: {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return size, source_hash.hexdigest()


def copy_temporal_ssl_run_to_drive(source_dir: str | Path,
                                   drive_dir: str | Path) -> Path:
    """Copy every run artifact and write a completion marker only after verification.

    The caller supplies a mounted Drive destination. A failed or interrupted copy
    has no completion marker; rerunning safely recopies the files.
    """
    source = Path(source_dir).resolve(strict=True)
    destination = Path(drive_dir).resolve(strict=False)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and Drive destination must be separate directories")
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("Run is incomplete; missing: " + ", ".join(missing))
    manifest = json.loads((source / "manifest_temporal_ssl.json").read_text(encoding="utf-8"))
    if manifest.get("mode") != "temporal_ssl" or not manifest.get("run_id"):
        raise ValueError("Source manifest is not a completed temporal SSL run")
    destination.mkdir(parents=True, exist_ok=True)
    existing_manifest = destination / "manifest_temporal_ssl.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("run_id") != manifest["run_id"]:
            raise FileExistsError("Drive destination already contains a different temporal SSL run")
    marker = destination / COMPLETION_MARKER
    if marker.exists():
        marker.unlink()
    copied: dict[str, dict[str, int | str]] = {}
    for file in sorted(source.rglob("*")):
        if not file.is_file() or file.name.endswith(".partial"):
            continue
        relative = file.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        size, digest = copy_verified_file(file, target)
        copied[relative.as_posix()] = {"bytes": size, "sha256": digest}
        print(f"Saved to Drive: {relative} ({size:,} bytes)")
    result = {
        "run_id": manifest["run_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(copied),
        "total_bytes": sum(item["bytes"] for item in copied.values()),
        "files": copied,
    }
    temporary_marker = marker.with_name(marker.name + ".partial")
    temporary_marker.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_marker, marker)
    return marker
