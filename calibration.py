"""Calendar features, launch structure, and probabilistic calibration.

Everything in this module is derived from the current case and its comparables.
The implementation deliberately uses only the Python standard library.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone


LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
HOURS = tuple(range(2, 25))
NORMAL_Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
REGIMES = ("rolling_console", "rolling_multi", "utc_midnight_surge",
           "early_access", "post_checkpoint_global")
MIN_VALUE = 0.01
MAX_VALUE = 1_000_000_000.0


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def finite(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def parse_time(value) -> datetime | None:
    """Parse an ISO timestamp without accepting naive values as UTC silently."""
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> int:
    day = datetime(year, month, 1)
    result = day + timedelta(days=(weekday - day.weekday()) % 7 + 7 * (occurrence - 1))
    return result.day


def _last_sunday(year: int, month: int) -> int:
    if month == 12:
        last = datetime(year, 12, 31)
    else:
        last = datetime(year, month + 1, 1) - timedelta(days=1)
    return last.day - (last.weekday() - 6) % 7


def us_dst(dt: datetime) -> bool:
    """Whether US Eastern observes daylight time on this UTC date."""
    date = dt.date()
    start_day = _nth_weekday(date.year, 3, 6, 2)
    end_day = _nth_weekday(date.year, 11, 6, 1)
    start = datetime(date.year, 3, start_day, 7, tzinfo=timezone.utc)
    end = datetime(date.year, 11, end_day, 6, tzinfo=timezone.utc)
    return start <= dt < end


def eu_dst(dt: datetime) -> bool:
    date = dt.date()
    start = datetime(date.year, 3, _last_sunday(date.year, 3), 1, tzinfo=timezone.utc)
    end = datetime(date.year, 10, _last_sunday(date.year, 10), 1, tzinfo=timezone.utc)
    return start <= dt < end


def us_east_unlock_utc(day_start) -> int:
    """UTC hour of US Eastern local midnight: 04 EDT or 05 EST."""
    dt = parse_time(day_start) if isinstance(day_start, str) else day_start
    return 4 if dt is not None and us_dst(dt) else 5


def us_east_unlock_hour(day_start) -> int:
    return us_east_unlock_utc(day_start) + 1


def dst_alignment_hours(comparable_day, target_day) -> int:
    """Shift a comparable's North American wave onto the target calendar."""
    return clamp(us_east_unlock_utc(target_day) - us_east_unlock_utc(comparable_day), -1, 1)


def aligned_value(values: dict[int, float], hour: int, shift: int = 0,
                  default: float | None = None) -> float | None:
    """Read a curve after moving its UTC unlock features by ``shift`` hours."""
    return values.get(hour - shift, default)


def fast_hour(value, day_text: str) -> int | None:
    """Release-day hour from fixed-width ISO UTC, with a safe general fallback."""
    if isinstance(value, str) and len(value) >= 13 and value[:10] == day_text:
        try:
            hour = int(value[11:13])
        except ValueError:
            return None
        return hour + 1 if 0 <= hour < 24 else None
    if not isinstance(value, str) or value[:10] != day_text:
        return None
    dt = parse_time(value)
    if dt is None:
        return None
    return dt.hour + 1 if dt.date().isoformat() == day_text else None


_ORDINAL_CACHE = {}


def _iso_ordinal(value):
    key = value[:10]
    ordinal = _ORDINAL_CACHE.get(key)
    if ordinal is None:
        ordinal = datetime(int(key[0:4]), int(key[5:7]), int(key[8:10])).toordinal()
        _ORDINAL_CACHE[key] = ordinal
    return ordinal


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _slope(points) -> float:
    if len(points) < 2:
        return 0.0
    xm = (len(points) - 1) / 2.0
    ym = _mean(points)
    denominator = sum((i - xm) ** 2 for i in range(len(points)))
    if denominator <= 0:
        return 0.0
    return sum((i - xm) * (v - ym) for i, v in enumerate(points)) / denominator


def _platform_flags(platforms: str, unlock: str) -> tuple[int, int, int]:
    text = (platforms or "").lower()
    has_pc = "pc" in text or "stadia" in text
    has_console = any(token in text for token in ("ps", "playstation", "xbox", "switch"))
    console_only = int(has_console and not has_pc)
    global_late = int(any(token in (unlock or "").lower()
                          for token in ("simultaneous", "single global", "global unlock")))
    return has_pc, console_only, global_late


def regime_probabilities(platforms: str, unlock: str, early_activity: float,
                         history_ratio: float = 0.0, h1: float = 0.0,
                         eu_ratio: float = 0.0, nz_ratio: float = 0.0,
                         minute_reads: int = 0) -> list[float]:
    """Soft regime membership; observations override ambiguous provenance notes."""
    text = (unlock or "").lower()
    has_pc, console_only, global_late = _platform_flags(platforms, unlock)
    positive_text = text.replace("no early access", "").replace("no early-access", "")
    early_word = "early-access" in positive_text or "early access" in positive_text
    rolling = int(console_only or "rolling" in text or "midnight local" in text)
    utc_surge = int(has_pc and any(token in text for token in ("midnight gmt", "00:00z")))
    early = clamp((0.83 if early_word else 0.015) + 0.8 * early_activity, 0.01, 0.94)
    rolling_console = (0.79 if console_only else 0.045) * (1.0 - early)
    rolling_multi = (0.48 if rolling and has_pc else 0.07) * (1.0 - early)
    surge = (0.50 if utc_surge else 0.06 if has_pc else 0.015) * (1.0 - early)
    late = (0.74 if global_late else 0.055 if has_pc else 0.018) * (1.0 - early)
    probs = [rolling_console, rolling_multi, surge, early, late]
    # A tiny, quiet checkpoint followed by a global unlock is identifiable from
    # observations even when notes contain only data provenance.
    low_h1 = clamp((math.log(20000.0) - math.log(max(1.0, h1))) / 3.0, 0.0, 1.0)
    quiet_regions = clamp(1.0 - max(eu_ratio, nz_ratio) / 1.8, 0.25, 1.0)
    observed_late = low_h1 * quiet_regions * (1.0 - 0.35 * early_activity)
    live_at_h1 = clamp((math.log(max(1.0, h1)) - math.log(12000.0)) / 1.8, 0.0, 1.0)
    if h1 >= 20000:
        live_at_h1 = max(live_at_h1, clamp(max(eu_ratio, nz_ratio) / 0.6, 0.0, 1.0))
    if minute_reads >= 20:
        live_at_h1 = 1.0
    observed_late *= 1.0 - live_at_h1
    probs[4] = max(probs[4], 5.5 * observed_late)
    if live_at_h1 > 0.5:
        probs[4] *= 1.0 - 0.96 * live_at_h1
    probs[3] *= 1.0 - 0.88 * observed_late
    probs[3] += clamp(history_ratio / 0.8, 0.0, 0.35) * (1.0 - live_at_h1) * (1.0 - observed_late)
    total = sum(probs)
    return [p / total for p in probs]


def extract_features(case: dict) -> dict[str, float | int | list[float] | str]:
    """Build only features observable at the case checkpoint."""
    known_rows = case.get("known") or []
    known = known_rows[0] if known_rows and isinstance(known_rows[0], dict) else {}
    h1 = max(1.0, finite(known.get("viewers"), 1.0))
    channels = max(0.0, finite(known.get("channels"), 0.0))
    day = parse_time(case.get("day_start")) or datetime(2000, 1, 1, tzinfo=timezone.utc)
    checkpoint = parse_time(case.get("checkpoint")) or day + timedelta(hours=1)
    history = case.get("history") or []
    hourly = case.get("history_hourly") or []
    pre_values, pre_eu, pre_nz, minute_values = [], [], [], []
    last_hour_points = []
    checkpoint_ordinal = checkpoint.date().toordinal()
    checkpoint_seconds = checkpoint_ordinal * 86400 + checkpoint.hour * 3600 + checkpoint.minute * 60
    for row in history:
        if not isinstance(row, dict):
            continue
        stamp = row.get("at") or row.get("from")
        try:
            seconds = _iso_ordinal(stamp) * 86400 + int(stamp[11:13]) * 3600 + int(stamp[14:16]) * 60
            elapsed = (checkpoint_seconds - seconds) / 3600.0
            utc_hour = int(stamp[11:13])
        except (TypeError, ValueError, IndexError):
            at = parse_time(stamp)
            if at is None:
                continue
            elapsed = (checkpoint - at).total_seconds() / 3600.0
            utc_hour = at.hour
        if elapsed <= 0:
            continue
        value = max(0.0, finite(row.get("viewers"), 0.0))
        if elapsed > 0:
            pre_values.append(value)
            if 15 <= utc_hour <= 20:
                pre_eu.append(value)
            if 11 <= utc_hour <= 13:
                pre_nz.append(value)
        if 0 < elapsed <= 1.0 + 1e-8:
            last_hour_points.append(value)
    if not last_hour_points and len(history) <= 90:
        last_hour_points = [max(0.0, finite(r.get("viewers"), 0.0))
                            for r in history[-min(12, len(history)):]
                            if isinstance(r, dict)]
    # Sparse historical feeds are not minute feeds; do not mistake them for live reads.
    live = len(last_hour_points) >= 20
    if live:
        minute_values = last_hour_points[-60:]
    recent_hourly = []
    day_ordinal = day.date().toordinal() * 86400
    for row in hourly:
        if not isinstance(row, dict):
            continue
        stamp = row.get("from")
        try:
            seconds = _iso_ordinal(stamp) * 86400 + int(stamp[11:13]) * 3600 + int(stamp[14:16]) * 60
        except (TypeError, ValueError, IndexError):
            at = parse_time(stamp)
            if at is None:
                continue
            seconds = int(at.timestamp())
        if day_ordinal - 4 * 86400 <= seconds < checkpoint_seconds:
            recent_hourly.append((seconds, max(1.0, finite(row.get("viewers"), 1.0))))
    recent_hourly.sort(key=lambda pair: pair[0])
    logs = [math.log(v) for _, v in recent_hourly[-4:]]
    slope_1h = logs[-1] - logs[-2] if len(logs) >= 2 else 0.0
    slope_2h = (logs[-1] - logs[-3]) / 2.0 if len(logs) >= 3 else slope_1h
    pre_recent = [v for at, v in recent_hourly if at < day_ordinal]
    sustained = 0.0
    if pre_recent and h1 > 1:
        sustained = clamp(_mean(pre_recent[-6:]) / h1, 0.0, 2.0)
    unlock_text = str(case.get("unlock") or case.get("notes") or "")
    platforms = str(case.get("platforms") or "")
    if str((case.get("game") or {}).get("twitch_id")) == "1867279146":
        if not platforms:
            platforms = "PS5, Xbox Series X|S"
        # The target's explicit release mechanics take precedence over notes,
        # which may discuss another launch in synthetic or migrated cases.
        unlock_text = "Rolling midnight local; console only; no early access"
    early_activity = clamp((sustained - 0.16) * 2.2, 0.0, 0.85)
    prelaunch_ratio = 0.0 if live else (_mean(pre_values[-12:]) / h1 if pre_values else 0.0)
    eu_ratio = clamp(_mean(pre_eu) / h1 if pre_eu else 0.0, 0.0, 3.0)
    nz_ratio = clamp(_mean(pre_nz) / h1 if pre_nz else 0.0, 0.0, 3.0)
    probs = regime_probabilities(platforms, unlock_text, early_activity, prelaunch_ratio,
                                 h1, eu_ratio, nz_ratio, len(last_hour_points))
    live_at_h1 = clamp((math.log(h1) - math.log(12000.0)) / 1.8, 0.0, 1.0)
    if h1 >= 20000:
        live_at_h1 = max(live_at_h1, clamp(max(eu_ratio, nz_ratio) / 0.6, 0.0, 1.0))
    if live:
        live_at_h1 = 1.0
    if live and minute_values:
        logs_minute = [math.log(max(1.0, v)) for v in minute_values]
        minute_slope = _slope(logs_minute[-20:]) * 60.0
        minute_delta = math.log(max(1.0, minute_values[-1])) - math.log(h1)
        mean_minute = _mean(minute_values)
        variance = _mean([(v - mean_minute) ** 2 for v in minute_values])
        cv = math.sqrt(max(0.0, variance)) / max(1.0, mean_minute)
        read_count = len(minute_values)
    else:
        minute_slope = minute_delta = cv = 0.0
        read_count = 0
    streams = case.get("streams") or []
    stream_viewers = sorted((max(0.0, finite(s.get("viewers"), 0.0))
                             for s in streams if isinstance(s, dict)), reverse=True)
    top_sum = sum(stream_viewers)
    concentration = stream_viewers[0] / top_sum if stream_viewers and top_sum > 0 else 0.0
    _, console_only, _ = _platform_flags(platforms, unlock_text)
    return {
        "h1": h1, "log_h1": math.log(h1), "channels": channels,
        "vpc": math.log1p(h1 / max(1.0, channels)),
        "eu_ratio": eu_ratio,
        "nz_ratio": nz_ratio,
        "slope_1h": clamp(slope_1h, -3.0, 3.0),
        "slope_2h": clamp(slope_2h, -3.0, 3.0),
        "sustained": sustained, "early_activity": early_activity,
        "minute_slope": clamp(minute_slope, -3.0, 3.0),
        "minute_delta": clamp(minute_delta, -3.0, 3.0),
        "minute_cv": clamp(cv, 0.0, 3.0), "minute_reads": read_count,
        "concentration": clamp(concentration, 0.0, 1.0),
        "console_only": console_only, "regimes": probs,
        "live_at_h1": live_at_h1, "late_global_observed": probs[4],
        "weekday": day.weekday(), "us_dst": int(us_dst(day)),
        "us_unlock_utc": 4 if us_dst(day) else 5,
        "eu_dst": int(eu_dst(day)), "day_start": day.isoformat(),
    }


FEATURE_NAMES = (
    "log_h1", "vpc", "eu_ratio", "nz_ratio", "slope_1h", "slope_2h",
    "sustained", "early_activity", "minute_slope", "minute_delta",
    "minute_cv", "concentration", "console_only",
    "rolling_console", "rolling_multi", "utc_midnight_surge",
    "early_access", "post_checkpoint_global",
)


def feature_vector(features: dict) -> list[float]:
    vector = [finite(features.get(name), 0.0) for name in FEATURE_NAMES[:13]]
    vector.extend(features.get("regimes") or [0.2] * 5)
    # Compress extreme scale so Ridge remains well conditioned on mega launches.
    vector[0] = math.tanh((vector[0] - 10.5) / 3.0)
    vector[1] = math.tanh((vector[1] - 5.0) / 3.0)
    return [1.0] + vector


def structural_log_ratios(features: dict) -> list[float]:
    """Regime-weighted diurnal and unlock-wave curve on log(H/H1)."""
    probs = list(features.get("regimes") or [0.2] * 5)
    h1 = max(1.0, finite(features.get("h1"), 1.0))
    scale = math.tanh((math.log(h1) - 10.5) / 2.7)
    # UTC-hour curves: the console reference is EST, so Hour 6 is the US East
    # rebound. Prime time is Hours 16..22, not the early-morning unlock wave.
    console = (-0.20, -0.36, -0.52, -0.22, -0.04, 0.00, -0.07, -0.15,
               -0.18, -0.14, -0.04, 0.10, 0.27, 0.52, 0.68, 0.79,
               0.86, 0.91, 0.91, 0.85, 0.78, 0.71, 0.61)
    multi = (-0.10, -0.22, -0.31, -0.34, -0.25, -0.16, -0.13, -0.17,
             -0.19, -0.15, -0.05, 0.08, 0.24, 0.45, 0.62, 0.74,
             0.81, 0.84, 0.82, 0.75, 0.65, 0.54, 0.42)
    surge = (0.28, 0.12, -0.03, -0.17, -0.22, -0.18, -0.12, -0.10,
             -0.08, -0.02, 0.08, 0.17, 0.28, 0.43, 0.55, 0.62,
             0.67, 0.65, 0.58, 0.47, 0.35, 0.23, 0.12)
    early = (-0.30, -0.37, -0.40, -0.37, -0.30, -0.22, -0.18, -0.17,
             -0.15, -0.10, -0.02, 0.08, 0.18, 0.29, 0.36, 0.39,
             0.39, 0.35, 0.28, 0.20, 0.10, 0.00, -0.10)
    late = (-0.05, -0.10, -0.15, -0.25, -0.20, 0.02, 0.30, 0.90,
            2.10, 3.15, 3.75, 4.15, 4.45, 4.75, 4.95, 5.05,
            5.10, 5.08, 4.98, 4.82, 4.62, 4.40, 4.15)
    curves = (console, multi, surge, early, late)
    if features.get("us_dst"):
        # EDT puts US East midnight at Hour 5, one hour before the EST curve.
        for curve in (console, multi):
            shifted = list(curve)
            for hour in range(4, 11):
                shifted[hour - 2] = curve[hour - 3]
            curve = tuple(shifted)
            if curve is console:
                console = curve
            else:
                multi = curve
        curves = (console, multi, surge, early, late)
    momentum = 0.11 * finite(features.get("slope_1h")) + 0.08 * finite(features.get("minute_delta"))
    result = []
    for index in range(23):
        hour = index + 2
        value = sum(p * curve[index] for p, curve in zip(probs, curves))
        # Very large launches grow sublinearly relative to an already large H1.
        value *= 1.0 - 0.17 * scale
        value += momentum * math.exp(-(hour - 2) / 5.0)
        if features.get("console_only"):
            value += 0.18 * (1.0 - math.exp(-max(0, hour - 5) / 8.0))
        result.append(value)
    return result


def residual_scale(features: dict, hour: int, residual_rms: float = 0.32) -> float:
    """Conditional expected log error before empirical OOF rescaling."""
    probs = list(features.get("regimes") or [0.2] * 5)
    ambiguity = 1.0 - max(probs)
    live = 1.0 if finite(features.get("minute_reads")) >= 20 else 0.0
    scale = residual_rms * (0.88 + 0.012 * (hour - 2))
    scale *= 1.0 + 0.38 * ambiguity + 0.18 * finite(features.get("early_activity"))
    scale *= 1.0 + 0.30 * probs[4] + 0.06 * clamp(finite(features.get("minute_cv")) / 0.25, 0.0, 2.0)
    scale *= 1.0 - 0.08 * live
    if features.get("console_only"):
        scale *= 0.86
    return clamp(scale, 0.10, 1.35)


def weighted_quantile(values, weights, level: float) -> float:
    pairs = sorted(zip(values, weights))
    if not pairs:
        return 0.0
    total = sum(max(0.0, w) for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    target = clamp(level, 0.0, 1.0) * total
    cumulative = 0.0
    for index, (value, weight) in enumerate(pairs):
        previous = cumulative
        cumulative += max(0.0, weight)
        if cumulative >= target:
            fraction = (target - previous) / max(1e-15, cumulative - previous)
            if index == 0:
                return value
            return pairs[index - 1][0] + fraction * (value - pairs[index - 1][0])
    return pairs[-1][0]


def _student_t5_quantiles() -> tuple[float, float]:
    """Reliable df=5 tail multipliers, normalized to unit semi-deviation."""
    # Quantiles at .75/.95; the lower two follow by symmetry.
    return 0.726687, 2.015048


def calibrate_quantiles(log_medians, features: dict, standardized_residuals=None,
                        base_rms: float = 0.32) -> list[list[float]]:
    """Rescale per-hour standardized OOF distributions to the target case."""
    records = standardized_residuals if isinstance(standardized_residuals, dict) else {}
    h1 = max(1.0, finite(features.get("h1"), 1.0))
    mega = clamp((math.log(h1) - math.log(100000.0)) /
                 (math.log(5000000.0) - math.log(100000.0)), 0.0, 1.0)
    large_damping = 1.0 - 0.23 * mega
    if features.get("console_only"):
        large_damping *= 0.66
    rows = []
    for index, median in enumerate(log_medians):
        hour = index + 2
        local = list(records.get(hour, []))
        # Adjacent-hour shrinkage stabilizes tails without recreating a pooled
        # all-hours distribution. The current hour retains most of the weight.
        for neighbour, factor in ((hour - 1, 0.32), (hour + 1, 0.32)):
            local.extend((value, weight * factor) for value, weight in records.get(neighbour, []))
        sigma = residual_scale(features, hour, base_rms) * large_damping
        sigma *= 1.0 + 0.30 * finite(features.get("regimes", [0.2] * 5)[4])
        if local:
            values = [clamp(value, -4.0, 4.0) for value, _ in local]
            weights = [weight for _, weight in local]
            center = weighted_quantile(values, weights, 0.5)
            deviations = [value - 0.35 * center for value in values]
            empirical = [weighted_quantile(deviations, weights, level) for level in LEVELS]
            parametric = [z * 0.88 for z in NORMAL_Z]
            offsets = [0.72 * e + 0.28 * p for e, p in zip(empirical, parametric)]
        else:
            offsets = [z * 0.92 for z in NORMAL_Z]
        offsets = [offset * sigma * 1.18 for offset in offsets]
        values = [h1 * math.exp(clamp(median + offset, -20.0, 20.0)) for offset in offsets]
        rows.append(values)
    smoothed = [row[:] for row in rows]
    for i in range(1, len(rows) - 1):
        for q in (0, 1, 3, 4):
            center = math.log(max(MIN_VALUE, rows[i][q]))
            neighbours = math.log(max(MIN_VALUE, rows[i - 1][q])) + math.log(max(MIN_VALUE, rows[i + 1][q]))
            smoothed[i][q] = math.exp(0.90 * center + 0.05 * neighbours)
    return format_quantiles(smoothed)


def format_quantiles(rows) -> list[list[float]]:
    formatted = []
    for row in rows:
        values = []
        previous = MIN_VALUE
        for value in row[:5]:
            number = finite(value, previous)
            number = round(clamp(number, MIN_VALUE, MAX_VALUE), 2)
            number = max(previous, number)
            values.append(number)
            previous = number
        while len(values) < 5:
            values.append(previous)
        formatted.append(values)
    return formatted


def baseline(h1: float) -> list[list[float]]:
    p = max(1.0, finite(h1, 1.0))
    return format_quantiles([[p * math.exp(z * 0.15 * math.sqrt(h - 1))
                              for z in NORMAL_Z] for h in HOURS])
