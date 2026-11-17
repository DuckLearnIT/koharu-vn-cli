"""Lightweight cross-platform desktop notifications (Windows Toast, macOS, Linux)."""

from __future__ import annotations

import base64
import html
import os
import re
import shutil
import subprocess
import sys


def _compact_text(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(1, limit - 1)].rstrip() + "…"


def _build_windows_toast_script(title: str, message: str, *, is_error: bool = False) -> str:
    safe_title = html.escape(_compact_text(title, 80))
    safe_msg = html.escape(_compact_text(message, 220))
    audio = (
        '<audio src="ms-winsoundevent:Notification.Reminder"/>'
        if is_error
        else '<audio src="ms-winsoundevent:Notification.Default"/>'
    )
    xml_payload = (
        "<toast>"
        "<visual><binding template=\"ToastGeneric\">"
        f"<text>{safe_title}</text>"
        f"<text>{safe_msg}</text>"
        "</binding></visual>"
        f"{audio}"
        "</toast>"
    )
    ps_escaped_xml = xml_payload.replace("'", "''")
    app_id = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
    return (
        "$ErrorActionPreference = 'SilentlyContinue'\n"
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null\n"
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null\n"
        "$xml = [Windows.Data.Xml.Dom.XmlDocument]::new()\n"
        f"$xml.LoadXml('{ps_escaped_xml}')\n"
        "$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)\n"
        f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{app_id}').Show($toast)\n"
    )


def send_desktop_notification(
    title: str,
    message: str,
    *,
    is_error: bool = False,
    enabled: bool = True,
) -> bool:
    """Fire-and-forget desktop notification when a job finishes or fails."""
    if not enabled:
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False

    short_title = _compact_text(title, 80)
    short_msg = _compact_text(message, 220)
    try:
        if sys.platform.startswith("win"):
            ps_script = _build_windows_toast_script(short_title, short_msg, is_error=is_error)
            encoded = base64.b64encode(ps_script.encode("utf-16le")).decode("ascii")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-WindowStyle",
                    "Hidden",
                    "-EncodedCommand",
                    encoded,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=flags,
            )
            return True

        if sys.platform == "darwin" and shutil.which("osascript"):
            esc_t = short_title.replace("\\", "\\\\").replace('"', '\\"')
            esc_m = short_msg.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.Popen(
                ["osascript", "-e", f'display notification "{esc_m}" with title "{esc_t}"'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True

        if shutil.which("notify-send"):
            urgency = "critical" if is_error else "normal"
            subprocess.Popen(
                ["notify-send", "-u", urgency, short_title, short_msg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
    except Exception:
        return False
    return False
