"""Skill installation from a remote index (opencode-style).

An index is a JSON document listing installable skills::

    {
        "version": "1",
        "skills": {
            "pdf": {
                "description": "Work with PDF files",
                "version": "1.2.0",
                "bundle": "https://example.com/skills/pdf-1.2.0.zip"
            }
        }
    }

``bundle`` may be a .zip or .tar.gz containing the skill folder (a directory
with a SKILL.md at its root, optionally nested one level deep).

Installs go through a staging directory and are moved into place atomically,
so a failed or interrupted download never leaves a half-written skill.
Installed versions are tracked in ``.clawd-version`` inside each skill dir,
so re-installing the same version is a no-op.
"""

from __future__ import annotations

import io
import json
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional


VERSION_FILE = ".clawd-version"
_DOWNLOAD_TIMEOUT = 60


class SkillInstallError(Exception):
    pass


def fetch_index(url: str) -> Dict[str, Any]:
    """Download and parse a skill index document."""
    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read()
    except Exception as exc:
        raise SkillInstallError(f"could not fetch skill index {url}: {exc}") from exc
    try:
        index = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise SkillInstallError(f"skill index {url} is not valid JSON: {exc}") from exc
    if not isinstance(index, dict) or not isinstance(index.get("skills"), dict):
        raise SkillInstallError(f"skill index {url} has no 'skills' object")
    return index


def list_index_skills(index: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten an index into a list of {name, description, version, bundle} dicts."""
    out = []
    for name, entry in index.get("skills", {}).items():
        if not isinstance(entry, dict):
            continue
        out.append(
            {
                "name": name,
                "description": entry.get("description", ""),
                "version": str(entry.get("version", "")),
                "bundle": entry.get("bundle", ""),
            }
        )
    return sorted(out, key=lambda e: e["name"])


def _safe_extract_zip(data: bytes, dest: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest.resolve()) + "/") and target != dest.resolve():
                raise SkillInstallError(f"unsafe path in bundle: {member.filename}")
        zf.extractall(dest)


def _safe_extract_tar(data: bytes, dest: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest.resolve()) + "/") and target != dest.resolve():
                raise SkillInstallError(f"unsafe path in bundle: {member.name}")
            if member.issym() or member.islnk():
                raise SkillInstallError(f"symlinks not allowed in skill bundle: {member.name}")
        tf.extractall(dest, filter="data")


def _find_skill_root(staging: Path) -> Path:
    """Locate the directory containing SKILL.md (top level or one nested)."""
    if (staging / "SKILL.md").is_file():
        return staging
    for child in sorted(staging.iterdir()):
        if child.is_dir() and (child / "SKILL.md").is_file():
            return child
    raise SkillInstallError("bundle does not contain a SKILL.md")


def _download(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as resp:
            return resp.read()
    except Exception as exc:
        raise SkillInstallError(f"could not download {url}: {exc}") from exc


def installed_version(skills_dir: Path, name: str) -> Optional[str]:
    vf = Path(skills_dir).expanduser() / name / VERSION_FILE
    if vf.is_file():
        return vf.read_text(encoding="utf-8").strip() or None
    return None


def install_skill(
    name: str,
    bundle_url: str,
    *,
    skills_dir: str | Path | None = None,
    version: str = "",
    force: bool = False,
) -> Path:
    """Download a skill bundle and install it atomically.

    Returns the installed skill directory. Re-installing the same version is
    a no-op unless ``force`` is true.
    """
    dest_root = Path(skills_dir).expanduser() if skills_dir else Path.home() / ".clawd" / "skills"
    dest = dest_root / name

    if not force and version and installed_version(dest_root, name) == version:
        return dest

    data = _download(bundle_url)
    lower = bundle_url.lower()
    with tempfile.TemporaryDirectory(prefix="clawd-skill-") as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir()
        if lower.endswith(".zip"):
            _safe_extract_zip(data, staging)
        elif lower.endswith((".tar.gz", ".tgz", ".tar")):
            _safe_extract_tar(data, staging)
        else:
            raise SkillInstallError(f"unsupported bundle type (want .zip or .tar.gz): {bundle_url}")
        skill_root = _find_skill_root(staging)

        final_staging = Path(tmp) / "final" / name
        final_staging.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(skill_root, final_staging)
        if version:
            (final_staging / VERSION_FILE).write_text(version, encoding="utf-8")

        dest_root.mkdir(parents=True, exist_ok=True)
        # Atomic swap: move new dir into place, replacing any old one.
        tmp_dest = dest_root / f".{name}.new"
        if tmp_dest.exists():
            shutil.rmtree(tmp_dest)
        shutil.move(str(final_staging), str(tmp_dest))
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(tmp_dest), str(dest))
    return dest


def install_from_index(
    index_url: str,
    name: str,
    *,
    skills_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Fetch an index and install the named skill from it."""
    index = fetch_index(index_url)
    for entry in list_index_skills(index):
        if entry["name"] == name:
            if not entry["bundle"]:
                raise SkillInstallError(f"index entry {name!r} has no bundle URL")
            return install_skill(
                name,
                entry["bundle"],
                skills_dir=skills_dir,
                version=entry["version"],
                force=force,
            )
    available = ", ".join(e["name"] for e in list_index_skills(index))
    raise SkillInstallError(f"skill {name!r} not in index (available: {available or 'none'})")
