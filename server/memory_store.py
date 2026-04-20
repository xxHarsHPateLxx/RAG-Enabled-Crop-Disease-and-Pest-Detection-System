from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class MemoryStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"sessions": {}}

        try:
            with self.path.open("r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
        except Exception:
            return {"sessions": {}}

        if not isinstance(data, dict):
            return {"sessions": {}}

        data.setdefault("sessions", {})
        return data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file_handle:
            json.dump(self._data, file_handle, indent=2, ensure_ascii=False)

    def ensure_session(self, session_id: str | None = None) -> str:
        resolved_session_id = session_id or str(uuid4())

        with self._lock:
            sessions = self._data.setdefault("sessions", {})
            sessions.setdefault(
                resolved_session_id,
                {
                    "created_at": self._now(),
                    "updated_at": self._now(),
                    "diagnoses": [],
                    "messages": [],
                },
            )
            self._save()

        return resolved_session_id

    def record_diagnosis(self, session_id: str, diagnosis: dict[str, Any]) -> str:
        diagnosis_id = diagnosis.get("diagnosis_id") or str(uuid4())
        entry = {
            **diagnosis,
            "diagnosis_id": diagnosis_id,
            "timestamp": self._now(),
        }

        with self._lock:
            session = self._data.setdefault("sessions", {}).setdefault(
                session_id,
                {
                    "created_at": self._now(),
                    "updated_at": self._now(),
                    "diagnoses": [],
                    "messages": [],
                },
            )
            session["diagnoses"].append(entry)
            session["updated_at"] = self._now()
            self._save()

        return diagnosis_id

    def record_message(self, session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            session = self._data.setdefault("sessions", {}).setdefault(
                session_id,
                {
                    "created_at": self._now(),
                    "updated_at": self._now(),
                    "diagnoses": [],
                    "messages": [],
                },
            )
            session["messages"].append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": self._now(),
                    "metadata": metadata or {},
                }
            )
            session["updated_at"] = self._now()
            self._save()

    def get_session(self, session_id: str) -> dict[str, Any]:
        session = self._data.get("sessions", {}).get(session_id)
        if not session:
            return {
                "session_id": session_id,
                "diagnoses": [],
                "messages": [],
            }

        return {
            "session_id": session_id,
            **session,
        }

    def recent_context(self, session_id: str, diagnosis_limit: int = 3, message_limit: int = 6) -> str:
        session = self.get_session(session_id)
        diagnoses = session.get("diagnoses", [])[-diagnosis_limit:]
        messages = session.get("messages", [])[-message_limit:]

        sections: list[str] = []

        if diagnoses:
            diagnosis_lines = []
            for diagnosis in diagnoses:
                diagnosis_lines.append(
                    f"- {diagnosis.get('timestamp', '')}: {diagnosis.get('crop', '')} / {diagnosis.get('disease', '')} (confidence {diagnosis.get('confidence', '')})"
                )
            sections.append("Recent diagnoses:\n" + "\n".join(diagnosis_lines))

        if messages:
            message_lines = []
            for message in messages:
                message_lines.append(f"- {message.get('role', '')}: {message.get('content', '')}")
            sections.append("Recent follow-up messages:\n" + "\n".join(message_lines))

        return "\n\n".join(sections).strip()
