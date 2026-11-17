"""Comparable-launch ensemble with honest leave-one-out calibration."""

from __future__ import annotations

import math

from calibration import (
    HOURS, aligned_value, baseline, calibrate_quantiles, clamp,
    dst_alignment_hours, extract_features, fast_hour, feature_vector, finite,
    parse_time, residual_scale, structural_log_ratios, weighted_quantile,
)


RIDGE_PENALTY = 5.0
HUBER_DELTA = 0.75


def _scan_comparable(comparable: dict):
    """One pass: release-day rows and the bounded pre-checkpoint feature history."""
    day_start = comparable.get("day_start")
    day_text = day_start[:10] if isinstance(day_start, str) else ""
    rows, history, hourly = {}, [], []
    checkpoint_prefix = day_text + "T00"
    source = comparable.get("hourly") or []
    # Complete SullyGnome windows place Hour 1 after 14 daily rows. Use the
    # timestamped release-day suffix directly; irregular inputs take the scan.
    start = 336 if len(source) > 336 and isinstance(source[336], dict) and str(source[336].get("from", "")).startswith(day_text + "T00") else None
    if start is None:
        start = next((i for i, row in enumerate(source)
                      if isinstance(row, dict) and str(row.get("from", "")).startswith(day_text + "T00")), None)
    recent_start = max(0, (start if start is not None else len(source)) - 96)
    for row in source[recent_start:]:
        if not isinstance(row, dict):
            continue
        stamp = row.get("from")
        hour = fast_hour(stamp, day_text)
        if hour is not None:
            rows[hour] = row
        if isinstance(stamp, str) and stamp[:13] <= checkpoint_prefix:
            history.append(row)
            hourly.append(row)
    # Comparable features use only the recent checkpoint window, not all 14 days.
    return rows, history[-96:], hourly[-96:]


def _release_rows(comparable: dict):
    return _scan_comparable(comparable)[0]


def _pseudo_case(comparable: dict, rows: dict) -> dict:
    _, history, hourly = _scan_comparable(comparable)
    h1 = rows[1]
    checkpoint = h1.get("from", "")[:13] + ":59:59Z"
    return {
        "id": "comparable", "day_start": comparable.get("day_start"),
        "checkpoint": checkpoint, "known": [dict(h1, hour=1)],
        "history": history, "history_hourly": hourly, "streams": None,
        "platforms": comparable.get("platforms"), "unlock": comparable.get("unlock"),
        "game": {"twitch_id": ""},
    }


def prepare_comparables(case: dict) -> list[dict]:
    prepared = []
    for comparable in case.get("comparables") or []:
        if not isinstance(comparable, dict):
            continue
        unlock = str(comparable.get("unlock") or "")
        if "target-history" in unlock or "not a launch" in unlock.lower():
            continue
        rows, history, hourly = _scan_comparable(comparable)
        if 1 not in rows or sum(h in rows for h in HOURS) < 12:
            continue
        h1 = max(1.0, finite(rows[1].get("viewers"), 1.0))
        if finite(rows[1].get("viewers"), 0.0) <= 0:
            continue
        pseudo = {
            "day_start": comparable.get("day_start"),
            "checkpoint": rows[1].get("from", "")[:13] + ":59:59Z",
            "known": [dict(rows[1], hour=1)], "history": history,
            "history_hourly": hourly, "streams": None,
            "platforms": comparable.get("platforms"), "unlock": unlock,
            "game": {"twitch_id": ""},
        }
        features = extract_features(pseudo)
        ratios = {h: math.log(max(1.0, finite(rows[h].get("viewers"), 1.0)) / h1)
                  for h in HOURS if h in rows}
        if len(ratios) < 12:
            continue
        prepared.append({
            "features": features, "x": feature_vector(features), "ratios": ratios,
            "day_start": comparable.get("day_start"), "unlock": unlock,
            "platforms": comparable.get("platforms", ""), "h1": h1,
        })
    return prepared


def _regression_vector(x):
    # Regime probabilities sum to one; dropping one avoids a singular dummy
    # block. Early-access activity is already represented by its regime mass.
    return x[:8] + x[9:18]


def _regime_weight(target_features, candidate_features):
    target_live = finite(target_features.get("live_at_h1"), 0.0)
    target_late = finite((target_features.get("regimes") or [0.2] * 5)[4])
    candidate_late = finite((candidate_features.get("regimes") or [0.2] * 5)[4])
    downweight = target_live * candidate_late
    mismatch = abs(target_late - candidate_late)
    return clamp((1.0 - 0.94 * downweight) * math.exp(-1.8 * mismatch), 0.025, 1.0)


def _matvec(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def _solve(matrix, vector):
    """Deterministic Gauss-Jordan solve for a small positive-definite system."""
    n = len(vector)
    augmented = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            augmented[column][column] += 1e-7
            pivot = column
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        if abs(scale) < 1e-20:
            continue
        augmented[column] = [v / scale for v in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[column])]
    return [augmented[i][-1] for i in range(n)]


def _inverse(matrix):
    """One Gauss-Jordan elimination with all identity right-hand sides."""
    n = len(matrix)
    work = [[matrix[i][j] + (1e-10 if i == j else 0.0) for j in range(n)] +
            [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(work[row][column]))
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        if abs(scale) < 1e-15:
            work[column][column] += 1e-9
            scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(n):
            if row == column:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [a - factor * b for a, b in zip(work[row], work[column])]
    return [row[n:] for row in work]


def _prepare_design(xs, weights):
    p = len(xs[0])
    means = [0.0] * p
    scales = [1.0] * p
    total_weight = sum(weights) or 1.0
    for j in range(1, p):
        means[j] = sum(w * x[j] for x, w in zip(xs, weights)) / total_weight
        variance = sum(w * (x[j] - means[j]) ** 2 for x, w in zip(xs, weights)) / total_weight
        scales[j] = max(0.12, math.sqrt(variance))
    standardized = [[1.0] + [(x[j] - means[j]) / scales[j] for j in range(1, p)] for x in xs]
    outer_products = [[[x[j] * x[k] for k in range(j, p)] for j in range(p)] for x in standardized]
    return standardized, outer_products, means, scales


def _weighted_fit(xs, ys, weights, penalty=RIDGE_PENALTY, prepared=None):
    if not xs:
        return [0.0], []
    if prepared is None:
        prepared = _prepare_design(xs, weights)
    standardized, outer_products, means, scales = prepared
    p = len(standardized[0])
    coefficients = [0.0] * p
    robust_weights = weights[:]
    for iteration in range(2):
        normal = [[0.0] * p for _ in range(p)]
        rhs = [0.0] * p
        robust_weights = []
        needs_robust = False
        for x, y, weight, outer in zip(standardized, ys, weights, outer_products):
            residual = y - sum(a * b for a, b in zip(x, coefficients))
            huber = min(1.0, HUBER_DELTA / max(HUBER_DELTA, abs(residual)))
            needs_robust = needs_robust or huber < 1.0
            w = max(1e-6, weight * huber)
            robust_weights.append(w)
            for j in range(p):
                rhs[j] += w * x[j] * y
                for k in range(j, p):
                    normal[j][k] += w * outer[j][k - j]
        for j in range(p):
            for k in range(j):
                normal[j][k] = normal[k][j]
            normal[j][j] += penalty if j else 1e-8
        updated = _solve(normal, rhs)
        if max(abs(a - b) for a, b in zip(updated, coefficients)) < 1e-7:
            coefficients = updated
            break
        coefficients = updated
        if iteration == 0 and not needs_robust:
            break
    inverse = _inverse(normal)
    raw = [0.0] * p
    for j in range(1, p):
        raw[j] = coefficients[j] / scales[j]
        raw[0] -= coefficients[j] * means[j] / scales[j]
    raw[0] += coefficients[0]
    press = []
    for x, y, weight in zip(standardized, ys, robust_weights):
        inverse_x = [sum(inverse[j][k] * x[j] for j in range(p)) for k in range(p)]
        leverage = weight * sum(a * b for a, b in zip(x, inverse_x))
        residual = y - sum(a * b for a, b in zip(x, coefficients))
        press.append(residual / max(0.25, 1.0 - leverage))
    return raw, press


def _similarity(target_x, candidate_x, candidate, target_features):
    squared = 0.0
    # Metadata/regime and observed momentum matter more than raw launch scale.
    feature_weights = (0.7, 0.2, 0.35, 0.35, 0.55, 0.45, 0.35, 0.5,
                       0.35, 0.35, 0.2, 0.2, 1.3, 1.5, 1.0, 0.9, 1.0, 0.8)
    for a, b, weight in zip(target_x[1:], candidate_x[1:], feature_weights):
        squared += weight * (a - b) ** 2
    if target_features.get("console_only") and candidate["features"].get("console_only"):
        squared *= 0.72
    distance = math.sqrt(squared / sum(feature_weights))
    early_penalty = 0.45 if candidate["features"].get("early_activity", 0.0) > 0.35 else 1.0
    return (math.exp(-1.35 * distance * distance) * early_penalty + 0.006) * _regime_weight(target_features, candidate["features"])


def _similarity_curves(target_features, target_x, comparables, target_day, fast=False):
    weights = [_similarity(target_x, c["x"], c, target_features) for c in comparables]
    target_unlock = finite(target_features.get("us_unlock_utc"), 5.0)
    aligned = []
    for c in comparables:
        shift = int(clamp(target_unlock - finite(c["features"].get("us_unlock_utc"), 5.0), -1, 1))
        cache_key = "aligned_edt" if target_unlock == 4 else "aligned_est"
        curve = c.get(cache_key)
        if curve is None:
            curve = {h: aligned_value(c["ratios"], h, shift) for h in HOURS}
            c[cache_key] = curve
        aligned.append(curve)
    curves = []
    scale = math.tanh((target_features["log_h1"] - 10.5) / 3.0)
    momentum = 0.06 * finite(target_features.get("slope_1h")) + 0.08 * finite(target_features.get("minute_delta"))
    for h in HOURS:
        pairs = [(curve[h], w) for curve, w in zip(aligned, weights) if curve.get(h) is not None]
        if not pairs:
            curves.append(0.0)
            continue
        values, local_weights = zip(*pairs)
        total = sum(local_weights)
        mean = sum(v * w for v, w in pairs) / total if total else 0.0
        median = mean if fast else weighted_quantile(values, local_weights, 0.5)
        value = mean if fast else 0.62 * median + 0.38 * mean
        value *= 1.0 - 0.10 * scale
        # Rolling launches commonly dip just before the North American wave.
        if target_features.get("console_only") and h in (3, 4, 5):
            value -= 0.035 * (6 - h)
        value += momentum * math.exp(-(h - 2) / 4.5)
        curves.append(value)
    return curves


def _ridge_models(train, target_features=None):
    models = {}
    design_cache = {}
    for hour in HOURS:
        usable = [c for c in train if hour in c["ratios"]]
        if len(usable) < 5:
            models[hour] = None
            continue
        key = tuple(id(c) for c in usable)
        cached = design_cache.get(key)
        if cached is None:
            xs = [_regression_vector(c["x"]) for c in usable]
            weights = [(0.75 + 0.25 / (1.0 + abs(c["features"]["log_h1"] - 10.5) / 5.0))
                       * (_regime_weight(target_features, c["features"]) if target_features else 1.0)
                       for c in usable]
            cached = (xs, weights, _prepare_design(xs, weights))
            design_cache[key] = cached
        xs, weights, prepared = cached
        ys = [c["ratios"][hour] for c in usable]
        coefficients, press = _weighted_fit(xs, ys, weights, prepared=prepared)
        center = weighted_quantile(ys, weights, 0.5)
        models[hour] = (usable, coefficients, press, center)
    return models


def _blend(ridge, similar, structural, train_size, features):
    late = finite(features.get("regimes", [0.2] * 5)[4])
    if late > 0.35:
        # Global-growth curves are structurally different from ordinary rolling
        # launches; contaminated analogues must not dominate their median.
        return 0.18 * ridge + 0.12 * similar + 0.70 * structural
    if ridge is None:
        ridge, wr = similar, 0.0
    else:
        wr = clamp((train_size - 5) / 35.0, 0.12, 0.48)
    ws = 0.37 if train_size else 0.0
    wp = max(0.12, 1.0 - wr - ws)
    if max(features.get("regimes") or [0.2]) > 0.65:
        wp += 0.20 if features.get("console_only") else 0.10
        ws = max(0.18, ws - 0.06)
        wr = max(0.0, wr - 0.04)
    return (wr * ridge + ws * similar + wp * structural) / (wr + ws + wp)


def _ensemble_predict(train, target_features, target_x, target_day, models=None):
    structural = structural_log_ratios(target_features)
    similar = _similarity_curves(target_features, target_x, train, target_day) if train else structural[:]
    models = _ridge_models(train) if models is None else models
    result = []
    for index, hour in enumerate(HOURS):
        model = models[hour]
        if model is None:
            ridge = None
        else:
            _, coefficients, _, center = model
            raw = sum(a * b for a, b in zip(coefficients, _regression_vector(target_x)))
            ridge = clamp(raw, center - 2.2, center + 2.2)
        result.append(_blend(ridge, similar[index], structural[index], len(train), target_features))
    return result


def _oof_standardized_residuals(comparables, models=None, target_features=None):
    """Per-hour OOF residuals standardized by each held-out launch's sigma."""
    records = {hour: [] for hour in HOURS}
    if len(comparables) < 7:
        return records
    target_regimes = (target_features or {}).get("regimes") or [0.2] * 5
    target_log_h1 = finite((target_features or {}).get("log_h1"), 10.5)
    models = models or _ridge_models(comparables, target_features)
    positions = {hour: {id(c): i for i, c in enumerate(model[0])}
                 for hour, model in models.items() if model is not None}
    for held_index, held in enumerate(comparables):
        train = comparables[:held_index] + comparables[held_index + 1:]
        structural = structural_log_ratios(held["features"])
        similar = _similarity_curves(held["features"], held["x"], train, held["day_start"], fast=True)
        regime_similarity = sum(a * b for a, b in zip(target_regimes, held["features"]["regimes"]))
        scale_distance = abs(target_log_h1 - held["features"]["log_h1"])
        weight = (0.05 + regime_similarity) * math.exp(-0.16 * scale_distance)
        for index, hour in enumerate(HOURS):
            actual = held["ratios"].get(hour)
            model = models[hour]
            if actual is None or model is None:
                continue
            usable, coefficients, press, center = model
            position = positions[hour].get(id(held), -1)
            if position < 0:
                continue
            fitted = sum(a * b for a, b in zip(coefficients, _regression_vector(held["x"])))
            ridge = clamp(fitted + press[position], center - 2.2, center + 2.2)
            prediction = _blend(ridge, similar[index], structural[index], len(train), held["features"])
            sigma = residual_scale(held["features"], hour, 0.34)
            standardized = clamp((actual - prediction) / sigma, -4.0, 4.0)
            records[hour].append((standardized, weight))
    return records


def _apply_oof_center(log_ratios, features, residual_records):
    if features.get("console_only") and finite(features.get("h1")) < 250000.0:
        return log_ratios
    adjusted = []
    for index, value in enumerate(log_ratios):
        records = residual_records.get(index + 2, []) if residual_records else []
        if records:
            values = [record[0] for record in records]
            weights = [record[1] for record in records]
            center = weighted_quantile(values, weights, 0.5)
            sigma = residual_scale(features, index + 2, 0.34)
            late = finite(features.get("regimes", [0.2] * 5)[4])
            strength = 0.78 + 0.16 * late
            adjustment = clamp(center * sigma * strength, -0.55, 0.75 + 2.2 * late)
        else:
            adjustment = 0.0
        adjusted.append(value + adjustment)
    return adjusted


def predict_prepared(comparables, features, day_start):
    """Shared by the entrypoint and external strict outer-LOO diagnostics."""
    if not comparables:
        return calibrate_quantiles(structural_log_ratios(features), features, [], 0.38)
    x = feature_vector(features)
    models = _ridge_models(comparables, features)
    log_ratios = _ensemble_predict(comparables, features, x, day_start, models)
    residuals = _oof_standardized_residuals(comparables, models, features)
    log_ratios = _apply_oof_center(log_ratios, features, residuals)
    return calibrate_quantiles(log_ratios, features, residuals, 0.34)


def forecast_case(case: dict) -> list[list[float]]:
    features = extract_features(case)
    comparables = prepare_comparables(case)
    return predict_prepared(comparables, features, case.get("day_start"))


def forecast(case: dict) -> dict:
    try:
        rows = forecast_case(case)
        if len(rows) != 23:
            raise ValueError("forecast must contain 23 rows")
    except Exception:
        known = case.get("known") or [{}]
        h1 = known[0].get("viewers", 1.0) if isinstance(known[0], dict) else 1.0
        rows = baseline(h1)
    return {"hours": [{"hour": hour, "q": row} for hour, row in zip(HOURS, rows)]}
