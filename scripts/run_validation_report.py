"""Generate the model-validation report under ``outputs/testing``.

Re-runs the deterministic validation suite (``src/validation/suite.py``) and
writes a JSON summary plus a compact Markdown table. Exits non-zero when any
check fails so CI can gate on the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.suite import run_validation_suite


def _markdown(report: dict) -> str:
    lines = [
        "# Model Validation Report",
        "",
        f"- Run date: `{report['run_date']}`",
        f"- Overall result: **{'PASS' if report['all_passed'] else 'FAIL'}**",
        "",
        "| Check | Status | Value | Threshold |",
        "|---|---|---|---|",
    ]
    for check in report["checks"]:
        value = json.dumps(check["value"], ensure_ascii=False)
        lines.append(
            f"| {check['name']} | {check['status']} | "
            f"{value} | {check['threshold']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    report = run_validation_suite()
    output_dir = PROJECT_ROOT / "outputs" / "testing"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "validation_report.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with (output_dir / "validation_report.md").open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(_markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved to {output_dir}")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
