"""GTA VI launch-day Twitch forecast: empirical quantiles from comparable console launches.

Model
-----
1. Parse every comparable launch in the case and keep only console-at-launch games
   (their release-day curve has the same rolling-midnight unlock shape as GTA VI).
2. For each kept launch, express release-day hours 2..24 as log-ratios to its own
   hour-1 print, so curves of different sizes share one scale.
3. For each forecast hour, pool the neighbouring hours' log-ratios (a small window),
   take their weighted mean and weighted standard deviation, and widen the spread by
   the sample-size inflation (1 + 1/n_eff) and a fixed model-risk floor.
4. Blend the Gaussian quantiles with the weighted empirical quantiles (80/20), which
   keeps the tails honest on a small sample while the Gaussian part smooths them.
5. Momentum correction: when hour 1 printed BELOW the pre-launch hour 0 (the EU wave
   had not landed yet), the US surge later in the day is systematically bigger in
   ratio terms, so hours 5..24 get a log-shift that ramps in with the hour.
6. Quantiles = hour-1 print * exp(log-quantile), rounded to 2 decimals (half up).

Everything is deterministic: no time reads, no randomness, sorted iteration only.
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

# Console-at-launch launches whose release-day curve we trust as a shape reference.
# The four practice launches of this competition are added because the final case's
# comparables hold them too, and they are the closest shapes to GTA VI's unlock.
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

# Shape parameters, fitted offline on the four practice cases (their answers come
# from SullyGnome, the same source the case files name).
SMOOTH_WINDOW = 2      # hours pooled around h for the mean and spread
GAUSS_BLEND = 0.8      # weight of the Gaussian quantiles over the empirical ones
SPREAD_K = 1.4         # multiplier on the sample spread
FLOOR = 0.12           # model-risk floor added in quadrature to every spread
BOOST_MULT = 1.8       # strength of the hour-1-below-hour-0 correction
BOOST_CAP = 1.2        # cap on that correction, in log units


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


def comparable_ratios(case):
    """Return [(name, weight, {hour: log-ratio to hour-1 print}), ...] for the
    console subset of the case's comparables."""
    rows = []
    for comp in case.get("comparables") or []:
        name = comp.get("name") or ""
        if not any(key in name for key in CONSOLE_NAMES):
            continue
        unlock = comp.get("unlock") or ""
        day_start = comp.get("day_start")
        if not day_start:
            continue
        try:
            year = int(day_start[:4])
        except ValueError:
            year = 2020
        weight = tier_weight(unlock) * recency_weight(year)
        day0 = parse_time(day_start)
        hourly = comp.get("hourly") or []
        by_hour = {}
        for point in hourly:
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
            rows.append((name, weight, ratios))
    return rows


def weighted_quantile(values, weights, level):
    """The interpolated weighted quantile of the sorted sample at `level`."""
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    target = level * total
    cumulative = 0.0
    for index, (value, weight) in enumerate(pairs):
        if cumulative + weight >= target:
            if index == 0 or weight <= 0:
                return value
            prev_value, _ = pairs[index - 1]
            fraction = (target - cumulative) / weight
            return prev_value + fraction * (value - prev_value)
        cumulative += weight
    return pairs[-1][0]


def forecast_case(case):
    first_row = next((row for row in case["known"] if row["hour"] == 1), None)
    if first_row is None:
        raise ValueError("case has no hour-1 print")
    hour1 = max(1.0, float(first_row["viewers"]))
    rows = comparable_ratios(case)
    if not rows:
        # A case whose comparables hold none of the console names: fall back to
        # every comparable so the forecast is still shape-based, never empty.
        rows = comparable_ratios_all(case)

    # Pool per-hour samples in a small window.
    pooled = {}
    for hour in range(2, 25):
        values = []
        weights = []
        for lo in range(max(2, hour - SMOOTH_WINDOW + 1), min(24, hour + SMOOTH_WINDOW) + 1):
            for _, weight, ratios in rows:
                if lo in ratios:
                    values.append(ratios[lo])
                    weights.append(weight)
        pooled[hour] = (values, weights)

    # Mean and spread per hour from the pooled samples.
    means = {}
    spreads = {}
    for hour in range(2, 25):
        values, weights = pooled[hour]
        if not values:
            means[hour] = 0.0
            spreads[hour] = 0.5
            continue
        total_weight = sum(weights)
        mean = sum(value * weight for value, weight in zip(values, weights)) / total_weight
        variance = sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights)) / total_weight
        means[hour] = mean
        spreads[hour] = math.sqrt(max(variance, 0.01))

    # Momentum correction: hour 1 below the pre-launch hour 0 means the EU wave had
    # not landed, so the US surge is bigger in ratio terms.
    boost = 0.0
    hour0 = None
    for point in case.get("history_hourly") or []:
        if str(point.get("from", "")).endswith("T23:00:00Z"):
            hour0 = point.get("viewers")
    if hour0 and hour0 > 0 and first_row["viewers"] > 0:
        ratio01 = first_row["viewers"] / hour0
        boost = min(BOOST_MULT * max(0.0, math.log(0.8) - math.log(ratio01)), BOOST_CAP)

    forecast = []
    for row in case["forecast"]:
        hour = row["hour"]
        values, weights = pooled[hour]
        if not values:
            quantiles = [hour1 * math.exp(z * 0.15 * math.sqrt(hour - 1)) for z in ZSCORES]
        else:
            total_weight = sum(weights)
            n_eff = total_weight * total_weight / sum(w * w for w in weights)
            mean = means[hour]
            if boost > 0 and hour >= 5:
                mean += boost * min(1.0, (hour - 4) / 6.0)
            spread = math.sqrt(
                spreads[hour] ** 2 * (1 + 1 / n_eff) * SPREAD_K + FLOOR ** 2
            )
            gaussian = [mean + z * spread for z in ZSCORES]
            empirical = [weighted_quantile(values, weights, level) for level in LEVELS]
            blended = [GAUSS_BLEND * g + (1 - GAUSS_BLEND) * e for g, e in zip(gaussian, empirical)]
            # Enforce non-decreasing quantiles after blending.
            previous = None
            ordered = []
            for value in blended:
                if previous is not None and value < previous:
                    value = previous
                ordered.append(value)
                previous = value
            quantiles = [hour1 * math.exp(value) for value in ordered]
        forecast.append({"hour": hour, "q": [round2(q) for q in quantiles]})
    return {"hours": forecast}


def comparable_ratios_all(case):
    """Fallback: every comparable, console or not, same shape as comparable_ratios."""
    rows = []
    for comp in case.get("comparables") or []:
        name = comp.get("name") or ""
        day_start = comp.get("day_start")
        if not day_start:
            continue
        try:
            year = int(day_start[:4])
        except ValueError:
            year = 2020
        weight = tier_weight(comp.get("unlock") or "") * recency_weight(year)
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
            rows.append((name, weight, ratios))
    return rows


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
