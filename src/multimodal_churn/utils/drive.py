"""Minimal Google Drive shortcut support for shared Colab datasets."""

from typing import Any


SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
FOLDER_MIME = "application/vnd.google-apps.folder"


def ensure_project_shortcut(
    drive_service: Any, target_folder_id: str, shortcut_name: str
) -> bool:
    """Create a My Drive shortcut to a shared folder once; return whether created."""
    target = drive_service.files().get(
        fileId=target_folder_id, fields="id,name,mimeType"
    ).execute()
    if target.get("mimeType") != FOLDER_MIME:
        raise ValueError(f"Drive item {target_folder_id} is not a folder.")

    page_token = None
    while True:
        list_args = {
            "q": "'root' in parents and mimeType = 'application/vnd.google-apps.shortcut' and trashed = false",
            "fields": "nextPageToken,files(id,name,shortcutDetails)",
            "pageSize": 1000,
        }
        if page_token:
            list_args["pageToken"] = page_token
        page = drive_service.files().list(**list_args).execute()
        if any(
            item.get("name") == shortcut_name
            and item.get("shortcutDetails", {}).get("targetId") == target_folder_id
            for item in page.get("files", [])
        ):
            return False
        page_token = page.get("nextPageToken")
        if not page_token:
            break

    drive_service.files().create(
        body={
            "name": shortcut_name,
            "mimeType": SHORTCUT_MIME,
            "shortcutDetails": {"targetId": target_folder_id},
        },
        fields="id,name,shortcutDetails",
    ).execute()
    return True
