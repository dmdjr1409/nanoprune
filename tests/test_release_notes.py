import importlib.util
import unittest
from pathlib import Path

from nanoprune import __version__

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "release_notes.py"


def load_script():
    spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestReleaseNotes(unittest.TestCase):
    def test_current_version_can_be_released(self):
        notes = load_script().release_notes(__version__)
        self.assertIn("### Added", notes)
        self.assertIn(f"/releases/download/v{__version__}/nanoprune-{__version__}-py3-none-any.whl", notes)
        self.assertNotIn(f"## {__version__}", notes)  # the heading is the release title

    def test_bad_versions_are_refused(self):
        module = load_script()
        for version in ("0.7", "v0.7.0", "99.0.0", "0.7.0; rm -rf ~"):
            with self.assertRaises(ValueError):
                module.release_notes(version)

    def test_changelog_sections(self):
        module = load_script()
        changelog = "# Changelog\n\n## 1.1.0 — date\n\nNew.\n\n## 1.0.0 — date\n\nOld.\n"
        self.assertEqual(module.changelog_section("1.1.0", changelog), "New.")
        self.assertEqual(module.changelog_section("1.0.0", changelog), "Old.")
        with self.assertRaises(ValueError):
            module.changelog_section("1.0", changelog)  # "1.0" is not "1.0.0"
        with self.assertRaises(ValueError):
            module.changelog_section("2.0.0", changelog + "## 2.0.0 — date\n\n")  # empty section


if __name__ == "__main__":
    unittest.main()
