"""GTA VI launch-day Twitch forecast: basis-smoothed console-launch day shapes.

Model
-----
1. Parse every comparable launch and keep only console-at-launch games (their
   release-day curve shares GTA VI's rolling-midnight unlock shape). The four
   practice launches of this competition are in that set too, so the final case
   (whose comparables hold them) leans on the closest shapes available.
2. Express each kept launch's release-day hours 2..24 as log-ratios to its own
   hour-1 print, then fit each curve on a small Fourier/polynomial basis. Pooling
   basis coefficients instead of raw hours denoises the per-hour means while the
   fitted curve keeps the day's shape (EU decay, US surge, evening peak).
3. The spread per hour is the weighted deviation of the comparables' curves from
   the pooled curve (windowed over neighbouring hours), widened by the sample-size
   inflation (1 + 1/n) and a fixed model-risk floor. Quantiles are Gaussian around
   the pooled mean on the log scale.
4. Momentum correction: when hour 1 printed BELOW the pre-launch hour 0 (the EU
   unlock wave had not landed yet), the later US surge is systematically bigger in
   ratio terms, so hours 5..24 get a log-shift that ramps in with the hour.
5. On the final case only, the minute-level history feeds a small early-hours
   momentum term: the log-slope of the last 30 minutes of hour 1 nudges hours 2..4.
6. Quantiles = hour-1 print * exp(log-quantile), rounded to 2 decimals (half up).

Deterministic: no time reads, no randomness, sorted iteration only.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader runs `python3 -P -s /bot/main.py`: put this file's directory back on
# sys.path first so any module shipped beside it imports, as the kit's starter does.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ZSCORES = [-1.6448536, -0.6744898, 0.0, 0.6744898, 1.6448536]

# Console-at-launch launches whose release-day curve we trust as a shape reference,
# the four practice launches of this competition included (the final case's
# comparables hold them; weight them up when present).
CONSOLE_NAMES = (
    "Ghost of Yōtei",
    "Saros",
    "The Legend of Zelda: Tears of the Kingdom",
    "FINAL FANTASY VII REBIRTH",
    "The Last of Us Part II",
    "Ghost of Tsushima",
    "Stellar Blade",
    "Death Stranding 2: On the Beach",
    "Marvel's Wolverine",
    "Red Dead Redemption 2",
    "God of War Ragnarök",
    "Marvel's Spider-Man 2",
)

# The competition's four practice launches: the house's own pick of the closest
# shapes, present in the final case's comparables, weighted up when found.
PRACTICE_LAUNCHES = (
    "Marvel's Wolverine",
    "Red Dead Redemption 2",
    "God of War Ragnarök",
    "Marvel's Spider-Man 2",
)
PRACTICE_WEIGHT = 1.4

# Mega-launch comparables, mixed unlock patterns, added at low weight for scale
# diversity: GTA VI's day will be the biggest in the set, so the largest launches'
# fuller curves earn a small voice in the pool.
EXTRA_NAMES = {
    "ELDEN RING": 0.5,
    "Cyberpunk 2077": 0.35,
}

# Shape parameters, fitted offline on the four practice cases (their answers come
# from SullyGnome, the same source the case files name).
SPREAD_K = 0.75      # multiplier on the pooled spread
FLOOR = 0.08         # model-risk floor added in quadrature to every spread
WINDOW = 2           # hours pooled around h for the spread estimate
BOOST_MULT = 2.4     # strength of the hour-1-below-hour-0 correction
BOOST_CAP = 1.2      # cap on that correction, in log units
MOMENTUM_K = 0.5     # final-only: early-hour nudge per log-unit of 30-min slope
MOMENTUM_CAP = 0.25  # cap on the final-only momentum nudge
KERNEL_SIGMA = 0.5   # launch-scale kernel: weight comparables by |ln p1_c - ln p1_case|
KERNEL_FLOOR = 0.15  # minimum kernel weight once the case is beyond the pool's range
EARLY_K = 0.8       # early-hours correction when hour 1 printed below hour 0
VC_K = -0.4         # streamer-concentration correction on the early hours
VC_REF = 4.0        # reference log viewers-per-channel at hour 1
VC_HMAX = 8         # last hour the concentration correction reaches

# Per-hour median bias of the kernel-weighted pooled curve on the four practice
# launches, shrunk 0.7x and leave-one-out validated: corrects the systematic shape
# errors (under-predicted US-morning surge at hours 4-8, over-predicted mid-day
# plateau at hours 10-16). Applied as a log shift on every hour.
HOUR_BIAS = {
    2: 0.022173, 3: 0.025056, 4: 0.052182, 5: 0.250215, 6: 0.150676,
    7: 0.152303, 8: 0.064828, 9: -0.016497, 10: -0.110284, 11: -0.195655,
    12: -0.281673, 13: -0.298175, 14: -0.267099, 15: -0.215024, 16: -0.190042,
    17: -0.059622, 18: 0.086636, 19: 0.034660, 20: -0.017143, 21: -0.036431,
    22: 0.000092, 23: 0.040946, 24: -0.008454,
}


def round2(value):
    """Round to 2 decimals, half up, the way JavaScript's Math.round(x*100)/100 does."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    result = whole / 100
    return int(result) if result.is_integer() else result


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def tier_weight(unlock):
    """Weight a comparable by its tier note in the case data."""
    if "early-access-contaminated" in unlock:
        return 0.55
    if "weak" in unlock:
        return 0.45
    return 1.0


def recency_weight(year):
    if year >= 2024:
        return 1.25
    if year >= 2021:
        return 1.0
    return 0.8


def comparable_ratios(case, console_only=True):
    """Return [(name, weight, {hour: log-ratio to hour-1 print}, ln_p1)] for
    comparables whose release-day data parses. With console_only, keep only
    CONSOLE_NAMES plus the low-weight mega launches; the practice launches get a
    weight-up. ln_p1 feeds the launch-scale kernel in forecast_case."""
    rows = []
    for comp in case.get("comparables") or []:
        name = comp.get("name") or ""
        is_console = any(key in name for key in CONSOLE_NAMES)
        is_extra = name in EXTRA_NAMES
        if console_only and not is_console and not is_extra:
            continue
        day_start = comp.get("day_start")
        if not day_start:
            continue
        try:
            year = int(day_start[:4])
        except ValueError:
            year = 2020
        weight = tier_weight(comp.get("unlock") or "") * recency_weight(year)
        if is_extra:
            weight *= EXTRA_NAMES[name]
        elif any(key in name for key in PRACTICE_LAUNCHES):
            weight *= PRACTICE_WEIGHT
        day0 = parse_time(day_start)
        by_hour = {}
        for point in comp.get("hourly") or []:
            try:
                at = parse_time(point["from"])
            except (KeyError, ValueError):
                continue
            viewers = point.get("viewers")
            if viewers is None:
                continue
            for hour in range(1, 25):
                if at == day0 + timedelta(hours=hour - 1):
                    by_hour[hour] = viewers
                    break
        first = by_hour.get(1)
        if not first or first <= 0:
            continue
        ratios = {}
        for hour in range(2, 25):
            value = by_hour.get(hour)
            if value is not None and value > 0:
                ratios[hour] = math.log(max(value / first, 1e-4))
        if ratios:
            rows.append((name, weight, ratios, math.log(first)))
    return rows


# --- basis machinery ---------------------------------------------------------

def basis_row(hour):
    x = hour / 24.0
    return [1.0, x, x * x, math.sin(2 * math.pi * x), math.cos(2 * math.pi * x),
            math.sin(4 * math.pi * x), math.cos(4 * math.pi * x), math.log(hour),
            math.exp(-hour / 3.0), 1.0 / hour,
            math.sin(6 * math.pi * x), math.cos(6 * math.pi * x)]


def fit_basis(ratios):
    """Least-squares basis coefficients of a comparable's log-ratio curve."""
    hours = sorted(h for h in ratios if 2 <= h <= 24)
    if len(hours) < 10:
        return None
    matrix = [basis_row(h) for h in hours]
    target = [ratios[h] for h in hours]
    size = len(matrix[0])
    normal = [[sum(matrix[i][j] * matrix[i][k] for i in range(len(matrix))) for k in range(size)]
              for j in range(size)]
    rhs = [sum(matrix[i][j] * target[i] for i in range(len(matrix))) for j in range(size)]
    augmented = [row[:] + [rhs[i]] for i, row in enumerate(normal)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(augmented[r][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        if abs(augmented[col][col]) < 1e-12:
            return None
        for row in range(col + 1, size):
            factor = augmented[row][col] / augmented[col][col]
            for c in range(col, size + 1):
                augmented[row][c] -= factor * augmented[col][c]
    coef = [0.0] * size
    for i in range(size - 1, -1, -1):
        coef[i] = (augmented[i][size] - sum(augmented[i][j] * coef[j] for j in range(i + 1, size))) / augmented[i][i]
    return coef


def curve_value(coef, hour):
    x = hour / 24.0
    return (coef[0] + coef[1] * x + coef[2] * x * x + coef[3] * math.sin(2 * math.pi * x)
            + coef[4] * math.cos(2 * math.pi * x) + coef[5] * math.sin(4 * math.pi * x)
            + coef[6] * math.cos(4 * math.pi * x) + coef[7] * math.log(hour)
            + coef[8] * math.exp(-hour / 3.0) + coef[9] / hour
            + coef[10] * math.sin(6 * math.pi * x) + coef[11] * math.cos(6 * math.pi * x))


# --- forecast -----------------------------------------------------------------

def hour_zero_print(case):
    """The pre-launch hour 0 print (23:00-00:00 UTC of release day), if the case has it."""
    value = None
    for point in case.get("history_hourly") or []:
        if str(point.get("from", "")).endswith("T23:00:00Z"):
            value = point.get("viewers")
    return value


def minute_slope(case):
    """Final-case only: log-slope over the last 30 minutes of hour 1, or None."""
    history = [point for point in case.get("history") or [] if point.get("viewers")]
    if len(history) < 10:
        return None
    try:
        first_at = parse_time(history[0]["at"])
        last_at = parse_time(history[-1]["at"])
    except (KeyError, ValueError):
        return None
    if (last_at - first_at).total_seconds() < 3600:
        # fewer than an hour of data: not the final's minute recording
        return None
    # keep points in the last 30 minutes
    cutoff = last_at - timedelta(minutes=30)
    recent = []
    for point in history:
        try:
            at = parse_time(point["at"])
        except (KeyError, ValueError):
            continue
        if at >= cutoff:
            recent.append((at, point["viewers"]))
    if len(recent) < 10:
        return None
    recent.sort(key=lambda pair: pair[0])
    span = (recent[-1][0] - recent[0][0]).total_seconds()
    if span < 300:
        return None
    x0 = recent[0][0].timestamp()
    n = len(recent)
    mean_t = sum((at.timestamp() - x0) for at, _ in recent) / n
    mean_v = sum(math.log(max(v, 1.0)) for _, v in recent) / n
    num = sum((at.timestamp() - x0 - mean_t) * (math.log(max(v, 1.0)) - mean_v) for at, v in recent)
    den = sum((at.timestamp() - x0 - mean_t) ** 2 for at, _ in recent)
    if den <= 0:
        return None
    return num / den  # log viewers per second


def minute_accel(case):
    """Final-case only: log-slope change between the [-30,-15] and [-15,0] minute
    windows of hour 1, in log per hour squared (scaled); None without minute data."""
    history = [point for point in case.get("history") or [] if point.get("viewers")]
    if len(history) < 20:
        return None
    try:
        last_at = parse_time(history[-1]["at"])
    except (KeyError, ValueError):
        return None
    windows = {}
    for label, (lo, hi) in (("early", (30, 15)), ("late", (15, 0))):
        lo_at = last_at - timedelta(minutes=lo)
        hi_at = last_at - timedelta(minutes=hi)
        pts = []
        for point in history:
            try:
                at = parse_time(point["at"])
            except (KeyError, ValueError):
                continue
            if lo_at <= at < hi_at:
                pts.append((at, point["viewers"]))
        if len(pts) < 5:
            return None
        pts.sort(key=lambda pair: pair[0])
        x0 = pts[0][0].timestamp()
        n = len(pts)
        mean_t = sum((at.timestamp() - x0) for at, _ in pts) / n
        mean_v = sum(math.log(max(v, 1.0)) for _, v in pts) / n
        num = sum((at.timestamp() - x0 - mean_t) * (math.log(max(v, 1.0)) - mean_v) for at, v in pts)
        den = sum((at.timestamp() - x0 - mean_t) ** 2 for at, _ in pts)
        if den <= 0:
            return None
        windows[label] = num / den  # log viewers per second
    if "early" not in windows or "late" not in windows:
        return None
    # slope change per 15 minutes, scaled to an hour: log per hour per hour
    return (windows["late"] - windows["early"]) * 900.0


def forecast_case(case):
    first_row = next((row for row in case["known"] if row["hour"] == 1), None)
    if first_row is None:
        raise ValueError("case has no hour-1 print")
    hour1 = max(1.0, float(first_row["viewers"]))
    rows = comparable_ratios(case, console_only=True)
    if not rows:
        rows = comparable_ratios(case, console_only=False)

    # launch-scale kernel: weight comparables by how close their hour-1 print is
    # to this case's (bigger launches have systematically smaller day ratios).
    # When the case's hour-1 print runs past the pool's range the raw kernel would
    # collapse every weight to ~zero (leaving the model an effectively empty pool),
    # so a floor takes over: the biggest launches plus a uniform share of the rest.
    case_ln_p1 = math.log(max(float(first_row["viewers"]), 1.0))
    pool_max = max((ln_p1 for _, _, _, ln_p1 in rows), default=case_ln_p1)
    kfloor = KERNEL_FLOOR if case_ln_p1 > pool_max + 0.5 else 0.0
    rows = [(name, weight * max(math.exp(-((ln_p1 - case_ln_p1) ** 2) / (2 * KERNEL_SIGMA ** 2)), kfloor), ratios)
            for name, weight, ratios, ln_p1 in rows]

    fits = []
    for name, weight, ratios in rows:
        coef = fit_basis(ratios)
        if coef is not None:
            fits.append((name, weight, coef, ratios))
    if not fits:
        # every comparable failed the basis fit: fall back to persistence spread
        for name, weight, ratios in rows:
            fits.append((name, weight, None, ratios))

    total_weight = sum(weight for _, weight, _, _ in fits)
    if total_weight <= 0:
        total_weight = 1.0
    pooled = [sum(weight * (coef[i] if coef else 0.0) for _, weight, coef, _ in fits) / total_weight
              for i in range(12)]
    mean_h = {h: curve_value(pooled, h) for h in range(2, 25)}

    # per-hour spread: comparable curves' deviation from the pooled curve, windowed
    deviations = {h: [] for h in range(2, 25)}
    for name, weight, coef, ratios in fits:
        for h in range(2, 25):
            if h in ratios:
                deviations[h].append((ratios[h] - mean_h[h], weight))
    spreads = {}
    for h in range(2, 25):
        pooled_devs = []
        for hh in range(max(2, h - WINDOW + 1), min(24, h + WINDOW) + 1):
            pooled_devs += deviations[hh]
        if not pooled_devs:
            spreads[h] = 0.5
            continue
        wsum = sum(w for _, w in pooled_devs)
        variance = sum(w * d * d for d, w in pooled_devs) / wsum if wsum > 0 else 0.25
        spreads[h] = math.sqrt(max(variance, 0.01))

    # momentum correction from the hour-1 vs hour-0 relationship
    boost = 0.0
    early_boost = 0.0
    hour0 = hour_zero_print(case)
    if hour0 and hour0 > 0 and first_row["viewers"] > 0:
        ratio01 = first_row["viewers"] / hour0
        anomaly = max(0.0, math.log(0.8) - math.log(ratio01))
        boost = min(BOOST_MULT * anomaly, BOOST_CAP)
        # the opposite effect on the earliest hours: a crashed hour 1 recovers
        early_boost = min(EARLY_K * anomaly, 0.5)

    # streamer-concentration correction: a category dominated by few big streamers
    # decays harder in the early hours once those streamers sign off
    vc_shift = 0.0
    if first_row.get("channels") and first_row["channels"] > 0 and first_row["viewers"] > 0:
        vc_shift = VC_K * (math.log(first_row["viewers"] / first_row["channels"]) - VC_REF)
        vc_shift = max(-0.6, min(0.3, vc_shift))

    # off-scale case: hour 1 far above anything the category saw pre-launch means
    # the shape is being extrapolated, so widen every spread a little
    pre_max = 0.0
    for point in case.get("history_hourly") or []:
        viewers = point.get("viewers")
        if viewers:
            pre_max = max(pre_max, float(viewers))
    widen = 1.2 if (pre_max > 0 and first_row["viewers"] > 20 * pre_max) else 1.0

    # final-only: intra-hour-1 momentum nudges hours 2..4
    slope = minute_slope(case)
    if slope is None:
        momentum = 0.0
    else:
        momentum = max(-MOMENTUM_CAP, min(MOMENTUM_CAP, MOMENTUM_K * slope * 3600.0))
    accel = minute_accel(case)
    if accel is None:
        accel_nudge = 0.0
    else:
        accel_nudge = max(-0.15, min(0.15, 0.35 * accel))

    n_comparables = max(len(fits), 1)
    forecast = []
    for row in case["forecast"]:
        hour = row["hour"]
        mean = mean_h[hour] + HOUR_BIAS.get(hour, 0.0)
        if boost > 0 and hour >= 5:
            mean += boost * min(1.0, (hour - 4) / 6.0)
        if early_boost > 0 and hour <= 4:
            mean += early_boost * max(0.0, 1.0 - (hour - 2) / 2.0)
        if vc_shift != 0.0 and hour <= VC_HMAX:
            mean += vc_shift * max(0.0, 1.0 - (hour - 2) / (VC_HMAX - 1.0))
        if momentum != 0.0 and hour <= 4:
            mean += momentum * max(0.0, 1.0 - (hour - 2) / 3.0)
        if accel_nudge != 0.0 and hour <= 4:
            mean += accel_nudge * max(0.0, 1.0 - (hour - 2) / 2.0)
        spread = math.sqrt((spreads[hour] * widen) ** 2 * (1 + 1 / n_comparables) * SPREAD_K + FLOOR ** 2)
        quantiles_log = [mean + z * spread for z in ZSCORES]
        previous = None
        ordered = []
        for value in quantiles_log:
            if previous is not None and value < previous:
                value = previous
            ordered.append(value)
            previous = value
        quantiles = [hour1 * math.exp(value) for value in ordered]
        forecast.append({"hour": hour, "q": [round2(q) for q in quantiles]})
    return {"hours": forecast}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels") != LEVELS:
            raise ValueError(f"{path.name} asks for levels {case.get('levels')}")
        result = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
