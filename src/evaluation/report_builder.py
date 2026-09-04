"""Markdown report builder for evaluation results."""

from __future__ import annotations

import json
from pathlib import Path


def build_markdown_report(
    title: str,
    sections: dict[str, dict],
) -> str:
    lines = [f"# {title}", ""]
    for name, payload in sections.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append("```json")
        lines.append(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def save_markdown_report(
    output_path: str | Path,
    title: str,
    sections: dict[str, dict],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_markdown_report(title, sections),
        encoding="utf-8",
    )
    return path
