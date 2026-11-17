"""Deterministic make.fun entrypoint (Python 3.13 standard library only)."""

import sys

sys.dont_write_bytecode = True

from pathlib import Path

_REPOSITORY = Path(__file__).resolve().parent
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

import json
import math
import os

from forecaster import forecast


OUTPUT_LIMIT = 65_536
HOURS = tuple(range(2, 25))


def _sanitize(document, case):
    """Return the strict output schema, replacing every invalid numeric row."""
    known = case.get("known") or [{}] if isinstance(case, dict) else [{}]
    first = known[0] if known and isinstance(known[0], dict) else {}
    try:
        h1 = min(1_000_000_000.0, max(1.0, float(first.get("viewers", 1.0))))
        if not math.isfinite(h1):
            h1 = 1.0
    except (TypeError, ValueError, OverflowError):
        h1 = 1.0
    z = (-1.6448536269514722, -0.6744897501960817, 0.0,
         0.6744897501960817, 1.6448536269514722)
    fallback = [[min(1_000_000_000.0,
                     round(h1 * math.exp(level * 0.15 * math.sqrt(hour - 1)), 2))
                 for level in z] for hour in HOURS]
    supplied = document.get("hours") if isinstance(document, dict) else None
    by_hour = {}
    if isinstance(supplied, list):
        for row in supplied:
            if isinstance(row, dict) and type(row.get("hour")) is int:
                by_hour[row["hour"]] = row.get("q")
    clean_rows = []
    for index, hour in enumerate(HOURS):
        source = by_hour.get(hour)
        clean = []
        if isinstance(source, list) and len(source) == 5:
            previous = 0.01
            valid = True
            for value in source:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    valid = False
                    break
                number = float(value)
                if not math.isfinite(number) or number <= 0.0 or number > 1_000_000_000.0:
                    valid = False
                    break
                number = max(previous, round(number, 2))
                clean.append(number)
                previous = number
            if not valid:
                clean = []
        clean_rows.append(clean or fallback[index])
    return {"hours": [{"hour": hour, "q": row} for hour, row in zip(HOURS, clean_rows)]}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_dir.glob("*.in"), key=lambda path: path.name)
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                case = json.load(handle)
            if not isinstance(case, dict):
                case = {}
            document = _sanitize(forecast(case), case)
        except Exception:
            document = _sanitize({}, {})
        payload = json.dumps(document, ensure_ascii=True, separators=(",", ":"),
                             allow_nan=False).encode("ascii")
        if len(payload) > OUTPUT_LIMIT:
            payload = json.dumps(_sanitize({}, case), separators=(",", ":"),
                                 allow_nan=False).encode("ascii")
        destination = output_dir / (path.stem + ".out")
        with destination.open("wb") as handle:
            handle.write(payload)


if __name__ == "__main__":
    main()
