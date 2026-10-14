"""Tests for src/skills/install.py — opencode-style skill index downloading."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.skills.install import (
    SkillInstallError,
    fetch_index,
    install_from_index,
    install_skill,
    installed_version,
    list_index_skills,
)


def _make_bundle(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


class SkillInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.skills_dir = self.root / "skills"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _bundle(self, name="demo", body="Do the thing."):
        return _make_bundle({f"{name}/SKILL.md": f"---\nname: {name}\ndescription: Demo\n---\n{body}\n"})

    def test_list_index_skills(self) -> None:
        index = {"skills": {"b": {"description": "B", "version": "1", "bundle": "http://x/b.zip"},
                            "a": {"description": "A", "version": "2", "bundle": "http://x/a.zip"}}}
        listed = list_index_skills(index)
        self.assertEqual([e["name"] for e in listed], ["a", "b"])
        self.assertEqual(listed[0]["version"], "2")

    def test_fetch_index_rejects_bad_json(self) -> None:
        with patch("src.skills.install.urllib.request.urlopen", side_effect=Exception("nope")):
            with self.assertRaises(SkillInstallError):
                fetch_index("http://example.com/index.json")

    def test_install_skill_zip(self) -> None:
        data = self._bundle("demo")
        with patch("src.skills.install._download", return_value=data):
            dest = install_skill("demo", "http://example.com/demo.zip",
                                 skills_dir=self.skills_dir, version="1.0")
        self.assertTrue((dest / "SKILL.md").is_file())
        self.assertEqual(installed_version(self.skills_dir, "demo"), "1.0")

    def test_install_same_version_is_noop(self) -> None:
        data = self._bundle("demo")
        with patch("src.skills.install._download", return_value=data) as dl:
            install_skill("demo", "http://example.com/demo.zip",
                          skills_dir=self.skills_dir, version="1.0")
            # Tamper, then reinstall same version — should NOT re-download/overwrite
            (self.skills_dir / "demo" / "SKILL.md").write_text("tampered", encoding="utf-8")
            install_skill("demo", "http://example.com/demo.zip",
                          skills_dir=self.skills_dir, version="1.0")
            self.assertEqual(dl.call_count, 1)
            self.assertEqual((self.skills_dir / "demo" / "SKILL.md").read_text(), "tampered")

    def test_install_rejects_path_traversal(self) -> None:
        data = _make_bundle({"../evil.md": "x", "demo/SKILL.md": "---\nname: demo\n---\n"})
        with patch("src.skills.install._download", return_value=data):
            with self.assertRaises(SkillInstallError):
                install_skill("demo", "http://example.com/demo.zip", skills_dir=self.skills_dir)
        self.assertFalse((self.skills_dir / "demo").exists())

    def test_install_rejects_bundle_without_skill_md(self) -> None:
        data = _make_bundle({"demo/readme.txt": "hi"})
        with patch("src.skills.install._download", return_value=data):
            with self.assertRaises(SkillInstallError):
                install_skill("demo", "http://example.com/demo.zip", skills_dir=self.skills_dir)

    def test_install_from_index(self) -> None:
        data = self._bundle("demo")
        index = {"skills": {"demo": {"description": "D", "version": "3.0",
                                     "bundle": "http://example.com/demo.zip"}}}
        with patch("src.skills.install.fetch_index", return_value=index), \
             patch("src.skills.install._download", return_value=data):
            dest = install_from_index("http://example.com/index.json", "demo",
                                      skills_dir=self.skills_dir)
        self.assertTrue((dest / "SKILL.md").is_file())
        self.assertEqual(installed_version(self.skills_dir, "demo"), "3.0")

    def test_install_from_index_unknown_name(self) -> None:
        index = {"skills": {"demo": {"description": "D", "version": "1",
                                     "bundle": "http://example.com/demo.zip"}}}
        with patch("src.skills.install.fetch_index", return_value=index):
            with self.assertRaises(SkillInstallError) as cm:
                install_from_index("http://example.com/index.json", "nope",
                                   skills_dir=self.skills_dir)
        self.assertIn("demo", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
