"""Drive shortcut creation uses the shared project root and is repeatable."""

import unittest
from unittest.mock import Mock

from multimodal_churn.utils.drive import FOLDER_MIME, SHORTCUT_MIME, ensure_project_shortcut


class DriveShortcutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.files = Mock()
        self.service = Mock()
        self.service.files.return_value = self.files
        self.files.get.return_value.execute.return_value = {
            "id": "project-root-id",
            "name": "Project Master Thesis",
            "mimeType": FOLDER_MIME,
        }

    def test_creates_shortcut_to_project_root(self) -> None:
        self.files.list.return_value.execute.return_value = {"files": []}

        created = ensure_project_shortcut(
            self.service, "project-root-id", "Project Master Thesis"
        )

        self.assertTrue(created)
        body = self.files.create.call_args.kwargs["body"]
        self.assertEqual(body["mimeType"], SHORTCUT_MIME)
        self.assertEqual(body["shortcutDetails"]["targetId"], "project-root-id")
        self.assertNotIn("parents", body)  # Drive places it in My Drive by default.

    def test_reuses_matching_shortcut(self) -> None:
        self.files.list.return_value.execute.return_value = {
            "files": [{
                "name": "Project Master Thesis",
                "shortcutDetails": {"targetId": "project-root-id"},
            }]
        }

        created = ensure_project_shortcut(
            self.service, "project-root-id", "Project Master Thesis"
        )

        self.assertFalse(created)
        self.files.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
