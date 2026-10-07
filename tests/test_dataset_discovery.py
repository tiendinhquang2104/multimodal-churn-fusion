"""Mounted Drive path selection without needing a Colab session."""

import tempfile
import unittest
from pathlib import Path

from multimodal_churn.utils.config import find_drive_dataset_paths


class DriveDatasetDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.my_drive = Path(self.temp_dir.name) / "MyDrive"
        self.my_drive.mkdir()
        self.config = {
            "dataset": {
                "relative_path": "02_hm_fashion",
                "archive": "dataset.zip",
                "folder_url": "https://drive.google.com/drive/folders/example",
            }
        }

    def test_finds_shortcut_in_my_drive(self) -> None:
        archive = self.my_drive / "02_hm_fashion" / "dataset.zip"
        archive.parent.mkdir()
        archive.touch()

        root, paths = find_drive_dataset_paths(self.config, self.my_drive)

        self.assertEqual(root, self.my_drive)
        self.assertEqual(paths["archive"], archive)

    def test_override_selects_custom_shortcut_parent(self) -> None:
        root = self.my_drive / "Shortcuts"
        archive = root / "02_hm_fashion" / "dataset.zip"
        archive.parent.mkdir(parents=True)
        archive.touch()

        selected, paths = find_drive_dataset_paths(self.config, self.my_drive, root)

        self.assertEqual(selected, root)
        self.assertEqual(paths["archive"], archive)

    def test_missing_archive_reports_share_url(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "drive.google.com"):
            find_drive_dataset_paths(self.config, self.my_drive)


if __name__ == "__main__":
    unittest.main()
