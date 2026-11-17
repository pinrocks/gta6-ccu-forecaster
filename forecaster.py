"""Comparable-launch ensemble with honest leave-one-out calibration."""

from __future__ import annotations

import math
from datetime import timedelta

from calibration import (
    HOURS, aligned_value, baseline, calibrate_quantiles, clamp,
    dst_alignment_hours, extract_features, feature_vector, finite,
    parse_time, structural_log_ratios, weighted_quantile,
)


RIDGE_PENALTY = 5.0
HUBER_DELTA = 0.75


def _release_rows(comparable: dict):
    day = parse_time(comparable.get("day_start"))
    if day is None:
        return {}
    rows = {}
    for row in comparable.get("hourly") or []:
        if not isinstance(row, dict):
            continue
        at = parse_time(row.get("from"))
        if at is None:
            continue
        offset = int(round((at - day).total_seconds() / 3600.0)) + 1
        if 1 <= offset <= 24 and abs((at - day).total_seconds() / 3600.0 - (offset - 1)) < 0.01:
            rows[offset] = row
    return rows


def _pseudo_case(comparable: dict, rows: dict) -> dict:
    day = parse_time(comparable.get("day_start"))
    h1 = rows[1]
    history = []
    hourly = []
    h1at = parse_time(h1.get("from"))
    checkpoint = (h1at + timedelta(hours=1)).isoformat() if h1at else None
    for row in comparable.get("hourly") or []:
        at = parse_time(row.get("from")) if isinstance(row, dict) else None
        h1at = parse_time(h1.get("from"))
        if at is None or h1at is None or at > h1at:
            continue
        history.append({"at": row.get("from"), "viewers": row.get("viewers"),
                        "channels": row.get("channels")})
        hourly.append({"from": row.get("from"), "viewers": row.get("viewers"),
                       "channels": row.get("channels"), "reads": 1})
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
        rows = _release_rows(comparable)
        if 1 not in rows or sum(h in rows for h in HOURS) < 12:
            continue
        h1 = max(1.0, finite(rows[1].get("viewers"), 1.0))
        if finite(rows[1].get("viewers"), 0.0) <= 0:
            continue
        pseudo = _pseudo_case(comparable, rows)
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


def _weighted_fit(xs, ys, weights, penalty=RIDGE_PENALTY):
    if not xs:
        return [0.0], [], [0.0]
    n, p = len(xs), len(xs[0])
    means = [0.0] * p
    scales = [1.0] * p
    total_weight = sum(weights) or 1.0
    for j in range(1, p):
        means[j] = sum(w * x[j] for x, w in zip(xs, weights)) / total_weight
        variance = sum(w * (x[j] - means[j]) ** 2 for x, w in zip(xs, weights)) / total_weight
        scales[j] = max(0.12, math.sqrt(variance))
    standardized = [[1.0] + [(x[j] - means[j]) / scales[j] for j in range(1, p)] for x in xs]
    coefficients = [0.0] * p
    for _ in range(7):
        normal = [[0.0] * p for _ in range(p)]
        rhs = [0.0] * p
        robust_weights = []
        for x, y, weight in zip(standardized, ys, weights):
            residual = y - sum(a * b for a, b in zip(x, coefficients))
            huber = min(1.0, HUBER_DELTA / max(HUBER_DELTA, abs(residual)))
            w = max(1e-6, weight * huber)
            robust_weights.append(w)
            for j in range(p):
                rhs[j] += w * x[j] * y
                for k in range(j, p):
                    normal[j][k] += w * x[j] * x[k]
        for j in range(p):
            for k in range(j):
                normal[j][k] = normal[k][j]
            if j:
                normal[j][j] += penalty
            else:
                normal[j][j] += 1e-8
        updated = _solve(normal, rhs)
        if max(abs(a - b) for a, b in zip(updated, coefficients)) < 1e-7:
            coefficients = updated
            break
        coefficients = updated
    # Unstandardize so the same coefficient vector accepts raw feature vectors.
    raw = [0.0] * p
    for j in range(1, p):
        raw[j] = coefficients[j] / scales[j]
        raw[0] -= coefficients[j] * means[j] / scales[j]
    raw[0] += coefficients[0]
    return raw, means, scales


def _press_residuals(xs, ys, weights, coefficients, penalty=RIDGE_PENALTY):
    """Ridge PRESS residuals; solve once with all RHS, not once per row."""
    if len(xs) < 3:
        return [y - sum(a * b for a, b in zip(x, coefficients)) for x, y in zip(xs, ys)]
    p = len(xs[0])
    normal = [[0.0] * p for _ in range(p)]
    for x, weight in zip(xs, weights):
        for j in range(p):
            for k in range(j, p):
                normal[j][k] += weight * x[j] * x[k]
    for j in range(p):
        for k in range(j):
            normal[j][k] = normal[k][j]
        normal[j][j] += penalty if j else 1e-8
    inverse = [_solve(normal, [1.0 if j == k else 0.0 for k in range(p)])
               for j in range(p)]
    residuals = []
    for x, y in zip(xs, ys):
        leverage = weight * sum(x[j] * inverse[j][k] * x[k]
                                for j in range(p) for k in range(p))
        residual = y - sum(a * b for a, b in zip(x, coefficients))
        residuals.append(residual / max(0.25, 1.0 - leverage))
    return residuals


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
    return math.exp(-1.35 * distance * distance) * early_penalty + 0.012


def _similarity_curves(target_features, target_x, comparables, target_day):
    weights = [_similarity(target_x, c["x"], c, target_features) for c in comparables]
    aligned = []
    for c in comparables:
        shift = int(dst_alignment_hours(c["day_start"], target_day))
        aligned.append({h: aligned_value(c["ratios"], h, shift) for h in HOURS})
    curves = []
    scale = math.tanh((target_features["log_h1"] - 10.5) / 3.0)
    momentum = 0.06 * finite(target_features.get("slope_1h")) + 0.08 * finite(target_features.get("minute_delta"))
    for h in HOURS:
        pairs = [(curve[h], w) for curve, w in zip(aligned, weights) if curve.get(h) is not None]
        if not pairs:
            curves.append(0.0)
            continue
        values, local_weights = zip(*pairs)
        median = weighted_quantile(values, local_weights, 0.5)
        total = sum(local_weights)
        mean = sum(v * w for v, w in pairs) / total if total else median
        value = 0.62 * median + 0.38 * mean
        value *= 1.0 - 0.10 * scale
        # Rolling launches commonly dip just before the North American wave.
        if target_features.get("console_only") and h in (3, 4, 5):
            value -= 0.035 * (6 - h)
        value += momentum * math.exp(-(h - 2) / 4.5)
        curves.append(value)
    return curves


def _ridge_models(train):
    models = {}
    for hour in HOURS:
        usable = [c for c in train if hour in c["ratios"]]
        if len(usable) < 5:
            models[hour] = None
            continue
        xs = [c["x"] for c in usable]
        ys = [c["ratios"][hour] for c in usable]
        weights = [0.75 + 0.25 / (1.0 + abs(c["features"]["log_h1"] - 10.5) / 5.0)
                   for c in usable]
        coefficients, _, _ = _weighted_fit(xs, ys, weights)
        press = _press_residuals(xs, ys, weights, coefficients)
        center = weighted_quantile(ys, weights, 0.5)
        models[hour] = (usable, coefficients, press, center)
    return models


def _blend(ridge, similar, structural, train_size, features):
    if ridge is None:
        ridge, wr = similar, 0.0
    else:
        wr = clamp((train_size - 5) / 35.0, 0.12, 0.48)
    ws = 0.37 if train_size else 0.0
    wp = max(0.12, 1.0 - wr - ws)
    if max(features.get("regimes") or [0.2]) > 0.65:
        wp += 0.10
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
            raw = sum(a * b for a, b in zip(coefficients, target_x))
            ridge = clamp(raw, center - 2.2, center + 2.2)
        result.append(_blend(ridge, similar[index], structural[index], len(train), target_features))
    return result


def _oof_standardized_residuals(comparables, models=None, target_features=None):
    if len(comparables) < 7:
        return []
    models = _ridge_models(comparables) if models is None else models
    target_log_h1 = finite((target_features or {}).get("log_h1"), 10.5)
    target_regimes = (target_features or {}).get("regimes") or [0.2] * 5
    records = []
    for held_index, held in enumerate(comparables):
        train = comparables[:held_index] + comparables[held_index + 1:]
        structural = structural_log_ratios(held["features"])
        similar = _similarity_curves(held["features"], held["x"], train, held["day_start"])
        # Tiny pre-launch H1 values are intrinsically explosive for late global
        # unlocks. A bounded scale correction removes that systematic asymmetry.
        low_scale = clamp((8.0 - held["features"]["log_h1"]) / 4.0, 0.0, 1.0)
        correction = low_scale * (0.35 + 5.2 * held["features"]["regimes"][4])
        for index, hour in enumerate(HOURS):
            actual = held["ratios"].get(hour)
            model = models[hour]
            if actual is None or model is None:
                continue
            usable, coefficients, press, center = model
            try:
                position = usable.index(held)
            except ValueError:
                continue
            fitted = sum(a * b for a, b in zip(coefficients, held["x"]))
            ridge = clamp(fitted + press[position], center - 2.2, center + 2.2)
            prediction = _blend(ridge, similar[index], structural[index], len(train), held["features"])
            prediction += correction * (0.55 + 0.45 * math.exp(-(hour - 2) / 12.0))
            regime_similarity = sum(a * b for a, b in zip(target_regimes, held["features"]["regimes"]))
            scale_distance = abs(target_log_h1 - held["features"]["log_h1"])
            weight = (0.08 + regime_similarity) * math.exp(-0.10 * scale_distance)
            records.append((clamp(actual - prediction, -5.0, 5.0), weight))
    return records


def _apply_oof_center(log_ratios, features, residual_records):
    if not residual_records:
        return log_ratios
    values = [record[0] for record in residual_records]
    weights = [record[1] for record in residual_records]
    center = weighted_quantile(values, weights, 0.5)
    # Partial pooling avoids turning noisy median residuals into overcorrection.
    adjustment = clamp(center * 0.55, -0.32, 1.8)
    return [value + adjustment * (0.75 + 0.25 * index / 22.0)
            for index, value in enumerate(log_ratios)]


def predict_prepared(comparables, features, day_start):
    """Shared by the entrypoint and external strict outer-LOO diagnostics."""
    if not comparables:
        return calibrate_quantiles(structural_log_ratios(features), features, [], 0.38)
    x = feature_vector(features)
    models = _ridge_models(comparables)
    log_ratios = _ensemble_predict(comparables, features, x, day_start, models)
    low_scale = clamp((8.0 - features["log_h1"]) / 4.0, 0.0, 1.0)
    correction = low_scale * (0.35 + 5.2 * features["regimes"][4])
    log_ratios = [value + correction * (0.55 + 0.45 * math.exp(-(hour - 2) / 12.0))
                  for value, hour in zip(log_ratios, HOURS)]
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
