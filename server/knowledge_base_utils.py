from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def _format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _format_value(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for key, nested_value in value.items():
            formatted = _format_value(nested_value)
            if formatted:
                parts.append(f"{key.replace('_', ' ').title()}: {formatted}")
        return "\n".join(parts)

    if isinstance(value, list):
        parts = []
        for item in value:
            formatted = _format_value(item)
            if formatted:
                parts.append(f"- {formatted}")
        return "\n".join(parts)

    return _format_scalar(value)


def _normalize_crops_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for crop in payload.get("crops", []):
        crop_name = crop.get("name") or crop.get("crop") or crop.get("id") or "Unknown Crop"
        for disease in crop.get("diseases", []):
            entries.append(
                {
                    "crop": crop_name,
                    "disease": disease.get("name") or disease.get("disease") or disease.get("id") or "Unknown Disease",
                    "description": disease.get("description", ""),
                    "symptoms": disease.get("symptoms", {}),
                    "causes": {
                        "pathogen": disease.get("pathogen"),
                        "favorable_conditions": disease.get("favorable_conditions"),
                        "spread_mechanism": disease.get("spread_mechanism"),
                        "yield_impact": disease.get("yield_impact"),
                        "resistant_varieties": disease.get("resistant_varieties"),
                        "severity": disease.get("severity"),
                        "tags": disease.get("tags"),
                    },
                    "treatment": disease.get("treatment", {}),
                    "prevention": disease.get("prevention", []),
                }
            )
    return entries


def load_knowledge_entries(kbase_path: Path) -> list[dict[str, Any]]:
    with kbase_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if "knowledge_base" in payload:
        return payload["knowledge_base"]

    if "crops" in payload:
        return _normalize_crops_payload(payload)

    raise ValueError("Unsupported knowledge base schema. Expected 'knowledge_base' or 'crops'.")


def entry_to_text(entry: dict[str, Any]) -> str:
    parts = [
        f"Crop: {_format_value(entry.get('crop'))}",
        f"Disease: {_format_value(entry.get('disease'))}",
    ]

    for field in ("description", "symptoms", "causes", "treatment", "prevention"):
        value = entry.get(field)
        if value:
            parts.append(f"{field.title()}: {_format_value(value)}")

    return "\n".join(parts)


def build_documents_from_knowledge_base(kbase_path: Path) -> list[Document]:
    entries = load_knowledge_entries(kbase_path)
    documents: list[Document] = []

    for entry in entries:
        metadata = {
            "crop": entry.get("crop"),
            "disease": entry.get("disease"),
        }
        documents.append(Document(page_content=entry_to_text(entry), metadata=metadata))

    return documents