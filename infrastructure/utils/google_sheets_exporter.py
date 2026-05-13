"""Google Sheets task exporter using a Google Apps Script webhook."""

from __future__ import annotations

import os

import requests

SHEETS_WEBHOOK_URL_ENV = "SHEETS_WEBHOOK_URL"
SHEETS_WEBHOOK_SECRET_ENV = "SHEETS_WEBHOOK_SECRET"
SHEETS_WEBHOOK_TIMEOUT_SECONDS = 10


def export_task_to_sheet(sheet_id: str, task: str, status: str, job_id: str) -> bool:
    """Append a task status row to Google Sheets through an Apps Script webhook."""
    print(
        f"[DEBUG] Exporting task. Sheet ID: {sheet_id}, Task: {task}, "
        f"Status: {status}, Job ID: {job_id}"
    )

    webhook_url = os.getenv(SHEETS_WEBHOOK_URL_ENV)
    webhook_secret = os.getenv(SHEETS_WEBHOOK_SECRET_ENV)

    if not webhook_url:
        print(f"[SheetsExporter] ❌ Missing env var: {SHEETS_WEBHOOK_URL_ENV}")
        return False

    if not webhook_secret:
        print(f"[SheetsExporter] ❌ Missing env var: {SHEETS_WEBHOOK_SECRET_ENV}")
        return False

    payload = {
        "secret": webhook_secret,
        "sheet_id": sheet_id,
        "job_id": job_id,
        "task": task,
        "status": status,
    }

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=SHEETS_WEBHOOK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[SheetsExporter] ❌ Failed to call Apps Script webhook: {exc}")
        return False

    try:
        result = response.json()
    except ValueError:
        print(
            "[SheetsExporter] ❌ Apps Script returned non-JSON response: "
            f"{response.text[:500]}"
        )
        return False

    if result.get("ok") is True:
        print(f"[SheetsExporter] ✅ Exported task: {task}")
        return True

    print(f"[SheetsExporter] ❌ Apps Script rejected export: {result}")
    return False
