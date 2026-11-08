#!/usr/bin/env python3
"""Download the official Sesame source archive without invoking Git.

Only downloads/extracts data; never executes upstream code. Uses the Python
standard library. Existing extracted destinations are not overwritten.
"""
import argparse
import hashlib
import json
import shutil
import stat
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(repository="dorianborian/sesame-robot"):
    name = repository.split("/")[-1]
    URL = f"https://codeload.github.com/{repository}/zip/refs/heads/main"
    ARCHIVE = ROOT / f"downloads/{name}-main.zip"
    DEST = ROOT / f"assets/upstream/{name}"
    report_name = "asset_manifest.json" if name == "sesame-robot" else name.replace("-", "_") + "_manifest.json"
    REPORT = ROOT / "reports" / report_name
    if DEST.exists():
        raise SystemExit(f"Refusing to overwrite existing assets: {DEST}")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        partial = ARCHIVE.with_suffix(".zip.partial")
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as out:
            total = 0
            threshold = 20 * 1024 * 1024
            for block in iter(lambda: response.read(1024 * 1024), b""):
                total += len(block)
                if total > 1024 ** 3:
                    raise RuntimeError("Unexpected archive size exceeds 1 GiB")
                out.write(block)
                if total >= threshold:
                    print(f"Downloaded {total / 1024 ** 2:.1f} MiB", flush=True)
                    threshold += 20 * 1024 * 1024
        partial.rename(ARCHIVE)
    files = []
    with zipfile.ZipFile(ARCHIVE) as z:
        members = z.infolist()
        if sum(m.file_size for m in members) > 2 * 1024 ** 3:
            raise RuntimeError("Unexpected extracted size exceeds 2 GiB")
        prefixes = {PurePosixPath(m.filename).parts[0] for m in members}
        if len(prefixes) != 1:
            raise RuntimeError("Expected one archive root")
        for m in members:
            name = PurePosixPath(m.filename)
            if name.is_absolute() or ".." in name.parts or stat.S_ISLNK(m.external_attr >> 16):
                raise RuntimeError(f"Unsafe archive entry: {m.filename}")
        for m in members:
            rel = PurePosixPath(m.filename).relative_to(next(iter(prefixes)))
            if not rel.parts:
                continue
            target = DEST.joinpath(*rel.parts)
            if m.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            files.append({"path": str(rel), "bytes": target.stat().st_size, "sha256": digest(target)})
        revision = z.comment.decode("utf-8", errors="replace")
    counts = Counter(Path(f["path"]).suffix.lower() or "<none>" for f in files)
    models = [f["path"] for f in files if Path(f["path"]).suffix.lower() in {".urdf", ".mjcf", ".usd", ".usda", ".usdc", ".xml"}]
    report = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_url": f"https://github.com/{repository}",
        "archive_url": URL,
        "archive_revision_comment": revision,
        "archive_path": str(ARCHIVE.relative_to(ROOT)),
        "archive_sha256": digest(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size,
        "destination": str(DEST.relative_to(ROOT)),
        "license": "Apache-2.0; see preserved upstream LICENSE",
        "file_count": len(files),
        "total_extracted_bytes": sum(f["bytes"] for f in files),
        "extension_counts": dict(sorted(counts.items())),
        "simulation_model_candidates": models,
        "files": files,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="dorianborian/sesame-robot",
                        choices=["dorianborian/sesame-robot", "lukehollis/sesame-ml"])
    main(parser.parse_args().repository)
