"""Output text generation helpers for build_sft.py.

The functions return natural expert-style answer variants and lightweight
generation statistics used by the main script.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _safe_get(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        value = row.get(key, default)
    else:
        try:
            value = row[key] if key in row.index else default
        except Exception:
            value = getattr(row, key, default)
    if value is None:
        return default
    try:
        if value != value:
            return default
    except Exception:
        pass
    return value


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _fmt_pct(probability: Any) -> str:
    value = _to_float(probability)
    if value is None:
        return "Unknown"
    return f"{value * 100:.2f}%"


def _fmt_num(value: Any, digits: int = 2, suffix: str = "") -> str:
    number = _to_float(value)
    if number is None:
        return "Unknown"
    return f"{number:.{digits}f}{suffix}"


def _fmt_compact(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "Unknown"
    return f"{number:g}"


def _yes_no(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "Unknown"
    return "Yes" if number > 0 else "No"


def risk_level_from_row(row: Dict[str, Any]) -> str:
    """Use only the original retrieved risk_level; never re-derive it from risk_prob."""
    level = _safe_get(row, "risk_level")
    if level is not None:
        text = str(level).strip().capitalize()
        if text in {"Low", "Medium", "High"}:
            return text
    return "Unknown"


def _risk_score_str(row: Dict[str, Any]) -> Optional[str]:
    v = _to_float(_safe_get(row, "risk_score"))
    return f"{v:.2f}" if v is not None else None


def _gen_stats(used_bridge: bool = False, mostly_low_mh: bool = False) -> Dict[str, bool]:
    return {
        "calibrated_bridge_used": bool(used_bridge),
        "mostly_low_medium_high": bool(mostly_low_mh),
    }


def _pick(rng: Any, values: List[str]) -> str:
    return values[int(rng.integers(0, len(values)))]


def _pick_static(values: List[str], seed_val: Any = None) -> str:
    """Deterministic pick using a numeric seed (no rng needed — for use in _grid_context)."""
    idx = int(abs(seed_val * 1000) if seed_val is not None else 0) % len(values)
    return values[idx]


def _risk_frame(level: str) -> str:
    frames = {
        "Low": "lower-priority screening result",
        "Medium": "watch-list result that deserves further due diligence",
        "High": "high-priority screening constraint",
    }
    return frames.get(level, "risk screening result")


NORMAL_OPENING_TEMPLATES = [
    "Based on the retrieved assessment, PFI {pfi} is classified as {level} bushfire and site-screening risk, with a retrieved probability of {pct} and a composite risk score of {score}.",
    "PFI {pfi} sits in the {level} risk band ({pct} retrieved probability; risk score {score}), which should be read as a {frame} for renewable-energy site selection.",
    "The retrieved record gives PFI {pfi} a {level} risk label, with calibrated probability around {pct} (risk score {score}).",
    "For PFI {pfi}, I would treat the result as {level} risk, with the retrieved probability sitting at {pct} and a risk score of {score}.",
    "PFI {pfi} comes back as {level} risk in the retrieved data (risk score {score}), so it should be handled as a {frame}.",
    "The risk lookup for PFI {pfi} returns a {level} classification, a probability of {pct}, and a composite risk score of {score}.",
    "For this property, the retrieved result is {level} risk, with probability around {pct} and a risk score of {score}.",
    "PFI {pfi} is not being estimated from memory here; the retrieved record labels it as {level} risk at {pct} (risk score {score}).",
    "The current retrieved assessment places PFI {pfi} in the {level} band (risk score {score}), which is a {frame}.",
    "For screening purposes, PFI {pfi} should be read as {level} risk based on the retrieved probability of {pct} and a composite risk score of {score}.",
]


REASON_OPENING_TEMPLATES = [
    "PFI {pfi} is rated {level} because the retrieved risk workflow assigns that final band, with probability {pct} and a risk score of {score}.",
    "The {level} classification for PFI {pfi} comes from the retrieved model output (risk score {score}), not from a free-form estimate.",
    "For PFI {pfi}, the retrieved probability is {pct}, the risk score is {score}, and the final risk label is {level}; the explanation should be read as interpretation of that retrieved result.",
    "The short reason is that PFI {pfi} has a retrieved {level} label (risk score {score}), and the supporting fields help explain why that label is plausible.",
    "PFI {pfi} falls into the {level} category in the retrieved record (risk score {score}), so the reasoning starts from that final risk band.",
    "The retrieved assessment places PFI {pfi} at {pct} with a risk score of {score} and a final label of {level}; the visible features provide context for that result.",
    "This property is treated as {level} risk because the retrieved record shows that label for PFI {pfi}, with a composite risk score of {score}.",
    "For PFI {pfi}, the risk band is {level} (risk score {score}); the relevant question is which retrieved indicators help explain that classification.",
    "The retrieved data classifies PFI {pfi} as {level} with a risk score of {score}, and the feature context gives the main technical reasons to watch.",
    "PFI {pfi}'s {level} rating (risk score {score}) should be understood as the combined site-screening result, with the retrieved features acting as supporting evidence.",
]


SHORT_OPENING_TEMPLATES = [
    "PFI {pfi}: {level} risk, retrieved probability {pct}, risk score {score}.",
    "{level} risk for PFI {pfi}; retrieved probability is about {pct} (risk score {score}).",
    "PFI {pfi} is in the {level} risk band ({pct}; risk score {score}).",
    "Short answer: PFI {pfi} is {level} risk, around {pct} with a composite risk score of {score}.",
    "PFI {pfi} comes back as {level}, with probability {pct} and a risk score of {score}.",
    "The retrieved result for PFI {pfi} is {level} risk ({pct}; risk score {score}).",
    "PFI {pfi}: {level} band, probability about {pct}, risk score {score}.",
    "For PFI {pfi}, the headline result is {level} risk at {pct} (risk score {score}).",
    "PFI {pfi} is currently labelled {level} in the retrieved data, with a composite risk score of {score}.",
    "Risk lookup result: PFI {pfi} is {level}, with retrieved probability {pct} and a risk score of {score}.",
]


BRIDGE_MILD_TO_HIGHER_TEMPLATES = [
    "Some visible indicators look mild, but the final retrieved label is still {level}. I would not override that label using one or two low-looking fields.",
    "The individual fields do not all point upward, but the combined assessment still places the parcel in the {level} band.",
    "A few feature values appear low, but the retrieved record shows {level} risk with probability {pct} (risk score {score}), so the safer interpretation is to follow the final label.",
    "This is a case where the feature picture is mixed: some fields look benign, while the overall retrieved result remains {level}.",
    "The visible indicators are not uniformly severe, but the final risk band is {level}; I would explain the mild fields as context rather than treating them as a contradiction.",
    "Even though some inputs look favourable, the combined site-screening result still comes back as {level}.",
    "The final label should carry more weight than any single visible feature, especially when the fields point in different directions.",
    "The property has some lower-looking indicators, but the retrieved assessment still treats it as {level}, so the answer should remain anchored to that result.",
    "This looks like a mixed-evidence case: some features reduce concern, but the final retrieved band remains {level}.",
    "The lower-looking fields are worth mentioning, but they do not replace the retrieved {level} classification.",
]


BRIDGE_HIGHER_TO_LOW_TEMPLATES = [
    "Some visible indicators look elevated, but the final retrieved label is Low. I would treat those fields as monitoring points rather than overriding the overall result.",
    "A few fields may look adverse, but the combined assessment still classifies the parcel as Low.",
    "This is a mixed-evidence case: some features deserve attention, but the retrieved final band remains Low.",
    "The higher-looking indicators do not automatically move the site out of Low risk because the final label reflects the combined assessment.",
    "I would not turn this into a higher-risk answer based on one field alone; the retrieved label remains Low.",
    "The elevated-looking features are useful context, but the overall result still reads as Low.",
    "Some inputs point upward, yet the final retrieved classification remains Low, so the answer should stay aligned with that label.",
    "This property has a few items to watch, but the retrieved assessment still places it in the Low band.",
    "The final Low label suggests the adverse-looking fields are not strong enough on their own to change the screening result.",
    "There are some cautionary signals, but the combined retrieved result is still Low risk.",
]


def _signal_direction(row: Dict[str, Any]) -> str:
    high = low = 0

    def bump(value: Any, low_threshold: float, high_threshold: float) -> None:
        nonlocal high, low
        number = _to_float(value)
        if number is None:
            return
        if number >= high_threshold:
            high += 1
        elif number <= low_threshold:
            low += 1

    is_prone = _to_float(_safe_get(row, "is_prone", 0.0)) or 0.0
    if is_prone >= 0.5:
        high += 1
    else:
        low += 1
    bump(_safe_get(row, "closest_facility_distance"), 1500.0, 5000.0)
    bump(_safe_get(row, "veg_area"), 0.1, 1.0)
    bump(_safe_get(row, "fire_count"), 0.1, 1.0)
    bump(_safe_get(row, "yrs_since_last_burn"), 1.0, 20.0)
    bump(_safe_get(row, "cur_rate"), 0.01, 0.10)

    if low >= 4 and high == 0:
        return "mostly_low"
    if high >= 3:
        return "mostly_high"
    return "mixed"


def _feature_driver_phrases(row: Dict[str, Any], include_low_support: bool = True) -> List[str]:
    drivers: List[str] = []

    fac5 = _to_float(_safe_get(row, "total_facilities_5km"))
    nearest = _to_float(_safe_get(row, "closest_facility_distance"))
    is_prone = _to_float(_safe_get(row, "is_prone"))
    veg = _to_float(_safe_get(row, "veg_area"))
    evc_mut_1 = _to_float(_safe_get(row, "evc_mut_1"))
    evc_bcs_0 = _to_float(_safe_get(row, "evc_bcs_0"))
    xgroup_2 = _to_float(_safe_get(row, "xgroup_2"))
    xgroup_7 = _to_float(_safe_get(row, "xgroup_7"))
    fire_count = _to_float(_safe_get(row, "fire_count"))
    yrs = _to_float(_safe_get(row, "yrs_since_last_burn"))
    project_capacity = _to_float(_safe_get(row, "P_project"))
    curtailment = _to_float(_safe_get(row, "curtailment"))
    cur_rate = _to_float(_safe_get(row, "cur_rate"))

    if is_prone is not None and is_prone >= 0.5:
        drivers.append("the parcel intersects a mapped bushfire-prone area, which is a strong prior hazard indicator")
    if nearest is not None and nearest > 4000:
        drivers.append(f"response accessibility is weaker because the nearest fire-response facility is about {nearest:.0f} m away")
    if fac5 is not None and fac5 <= 1:
        drivers.append("there are limited fire-response facilities within the 5 km screening radius")
    if veg is not None and veg > 1:
        drivers.append(f"native vegetation cover is approximately {veg:g}, indicating significant available fuel")
    if evc_bcs_0 is not None and evc_bcs_0 >= 0.5:
        drivers.append("the EVC conservation-status flag indicates endangered or high-value vegetation, which can imply complex fuel structure")
    if xgroup_2 is not None and xgroup_2 >= 0.5:
        drivers.append("the macro vegetation group corresponds to Lower Slopes or Hills Woodlands, where slope and wind can accelerate spread")
    if xgroup_7 is not None and xgroup_7 >= 0.5:
        drivers.append("the riparian or swampy scrub woodland class can be moisture-buffered in normal periods but can deteriorate sharply under drought")
    if fire_count is not None and fire_count >= 1:
        fires = int(fire_count)
        drivers.append(f"the area has {fires} recorded fire {'event' if fires == 1 else 'events'} historically, indicating past fire activity")
    if yrs is not None and yrs >= 20:
        drivers.append(f"the long interval since last burn ({yrs:g} years) can indicate accumulated fuel load")
    if cur_rate is not None and cur_rate >= 0.10:
        drivers.append(f"the curtailment rate is elevated ({cur_rate:.4f}%), pointing to grid-export constraint pressure")
    elif curtailment is not None and curtailment > 10000:
        drivers.append(f"absolute curtailed energy is notable ({curtailment:.0f} MWh), indicating possible connection or dispatch constraints")

    if drivers or not include_low_support:
        return drivers

    if is_prone is not None and is_prone < 0.5:
        drivers.append("the property is not flagged as intersecting a mapped bushfire-prone area")
    if fac5 is not None and fac5 >= 2:
        drivers.append(f"response accessibility is supported by {int(fac5)} fire-response facilities within 5 km")
    if nearest is not None and nearest <= 2000:
        drivers.append(f"the nearest fire-response facility is comparatively close at about {nearest:.0f} m")
    if veg is not None and veg <= 1:
        drivers.append("the retrieved vegetation-fuel indicator is relatively mild")
    if fire_count is not None and fire_count <= 0:
        drivers.append("the retrieved record shows no historical nearby fire events")
    if cur_rate is not None and cur_rate < 0.10:
        drivers.append("the baseline curtailment rate appears low, suggesting limited current grid-export pressure")
    if evc_mut_1 is not None and evc_mut_1 >= 0.5:
        drivers.append("the mosaic vegetation structure can introduce fuel discontinuity, which may slow spread")
    return drivers or ["the retrieved indicators do not point strongly in a single direction"]


def _sample_drivers(row: Dict[str, Any], rng: Any, limit: int = 3, include_low_support: bool = True) -> List[str]:
    drivers = _feature_driver_phrases(row, include_low_support=include_low_support)
    if len(drivers) <= limit:
        return drivers
    order = list(drivers)
    rng.shuffle(order)
    return order[:limit]


def _join_sentence(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        text = items[0]
    else:
        text = "; ".join(items[:-1]) + "; and " + items[-1]
    return text[0].upper() + text[1:] + "."


# ---------------------------------------------------------------------------
# Contradiction-inference templates and helper
# Fires when surface fuel/fire indicators look mild but risk is still High/Medium,
# explaining which structural factor (is_prone, poor access) is the likely driver.
# ---------------------------------------------------------------------------

_PRONE_EXPLAINS_HIGH = [
    "Although the surface fuel metrics here appear relatively modest, this site sits within a mapped "
    "bushfire-prone area. Prone-area designations factor in landscape-scale hazards — terrain, wind "
    "exposure, and historical ignition patterns — that vegetation snapshots alone cannot capture. "
    "This zonal classification is most likely what keeps the site in the {level} risk band.",

    "Despite the limited vegetation coverage and low fire history, the bushfire-prone area overlay is "
    "a significant prior constraint. That designation reflects structural and landscape factors that can "
    "override benign fuel readings in the site-screening model, sustaining the {level} classification.",

    "The apparent gap between mild fuel indicators and a {level} risk result is most likely explained by "
    "the bushfire-prone area designation. This is a zone-level classification that encompasses risk factors "
    "beyond immediate fuel load, and the screening model gives it substantial weight.",
]

_ACCESS_EXPLAINS_HIGH = [
    "Even with limited native vegetation and a low fire history at this site, there is a meaningful "
    "structural constraint: {access_note}. In any ignition event, suppression response time is as "
    "critical as fuel load — a constrained access picture can keep a site in the {level} band even "
    "when the surface fuel indicators look mild.",

    "The elevated {level} classification here is not primarily about current fuel load. {access_note_cap}, "
    "which means that if an ignition event occurs, suppression resources may face significant delays in "
    "reaching the site. The screening model treats this response-time constraint as a material risk amplifier.",

    "This is a case where the structural risk factors outweigh the surface fuel picture. "
    "{access_note_cap}, and in a fast-moving fire scenario that kind of access gap can be decisive. "
    "The model likely weights this emergency-response constraint heavily, which explains the {level} result "
    "despite the lower-looking vegetation and fire history metrics.",
]

_PRONE_AND_ACCESS_EXPLAIN_HIGH = [
    "Two structural factors appear to be driving the {level} classification despite the limited fuel "
    "indicators. First, this parcel is mapped as a bushfire-prone area, reflecting landscape-scale "
    "hazard characteristics. Second, fire-response access is constrained: {access_note}. Either "
    "factor alone could sustain a {level} result; together they make the elevated classification "
    "straightforward to interpret even when vegetation and fire history look mild.",

    "The {level} result here is best understood as reflecting site structure rather than current fuel "
    "load. The property sits within a bushfire-prone area overlay, and emergency response access is "
    "limited ({access_note}). Both elements carry significant weight in the screening model independent "
    "of vegetation metrics, which is why the final classification remains elevated.",

    "Even with modest surface fuel readings, the site faces two converging structural constraints: "
    "the bushfire-prone area overlay and limited emergency response access ({access_note}). "
    "In combination, these factors explain why the model returns a {level} risk result — "
    "the hazard picture here is driven by landscape-level exposure and response capacity, "
    "not just by what the vegetation cover indicator shows.",
]


def _inferred_driver_note(row: Dict[str, Any], rng: Any) -> Optional[str]:
    """
    Returns a natural-language explanation when surface fire indicators are mild
    (low veg, no fire history) but the risk is still High or Medium, by identifying
    the structural driver: is_prone overlay and/or poor emergency-response access.
    """
    level = risk_level_from_row(row)
    if level not in {"High", "Medium"}:
        return None

    veg = _to_float(_safe_get(row, "veg_area"))
    fc = _to_float(_safe_get(row, "fire_count"))
    is_prone = _to_float(_safe_get(row, "is_prone"))
    fac5 = _to_float(_safe_get(row, "total_facilities_5km"))
    nearest = _to_float(_safe_get(row, "closest_facility_distance"))

    # Only fire when at least one surface fuel/fire indicator looks mild
    mild_surface = (veg is not None and veg < 1.0) or (fc is not None and fc == 0)
    if not mild_surface:
        return None

    # Identify structural drivers
    has_prone = is_prone is not None and is_prone >= 0.5

    access_note: Optional[str] = None
    if fac5 is not None and fac5 == 0:
        access_note = "there are no fire-response facilities within 5 km of this site"
    elif fac5 is not None and fac5 <= 1 and nearest is not None and nearest > 2500:
        fac_word = "facility" if int(fac5) == 1 else "facilities"
        access_note = (
            f"there is only {int(fac5)} fire-response {fac_word} within 5 km, "
            f"with the nearest approximately {nearest:.0f} m away"
        )
    elif nearest is not None and nearest > 4000:
        access_note = f"the nearest fire-response facility is approximately {nearest:.0f} m away"

    has_access_issue = access_note is not None

    if not has_prone and not has_access_issue:
        return None

    if has_prone and has_access_issue:
        return _pick(rng, _PRONE_AND_ACCESS_EXPLAIN_HIGH).format(
            level=level, access_note=access_note
        )
    if has_prone:
        return _pick(rng, _PRONE_EXPLAINS_HIGH).format(level=level)
    # access issue only
    access_note_cap = access_note[0].upper() + access_note[1:]
    return _pick(rng, _ACCESS_EXPLAINS_HIGH).format(
        level=level,
        access_note=access_note,
        access_note_cap=access_note_cap,
    )


def _bridge_explanation(row: Dict[str, Any], rng: Any) -> Optional[str]:
    level = risk_level_from_row(row)
    prob = _safe_get(row, "risk_probability", 0.0)
    pct = _fmt_pct(prob)
    score = _risk_score_str(row) or "N/A"
    direction = _signal_direction(row)
    if level in {"Medium", "High"} and direction == "mostly_low":
        return _pick(rng, BRIDGE_MILD_TO_HIGHER_TEMPLATES).format(level=level, pct=pct, score=score)
    if level == "Low" and direction == "mostly_high":
        return _pick(rng, BRIDGE_HIGHER_TO_LOW_TEMPLATES).format(level=level, pct=pct, score=score)
    return None


def _grid_context(row: Dict[str, Any]) -> Optional[str]:
    capacity = _to_float(_safe_get(row, "P_project"))
    curtailment = _to_float(_safe_get(row, "curtailment"))
    cur_rate = _to_float(_safe_get(row, "cur_rate"))
    if capacity is None and curtailment is None and cur_rate is None:
        return None
    sentences: List[str] = []
    if capacity is not None:
        sentences.append(
            f"If you are planning to build a renewable energy project at this site, "
            f"the recommended installed capacity is {_fmt_num(capacity, 2)} MW."
        )
    if curtailment is not None and cur_rate is not None:
        sentences.append(
            f"Based on the zone substation load and grid import/export headroom data for this area, "
            f"the inferred curtailment during the assessment period is approximately {_fmt_num(curtailment, 2)} MWh, "
            f"corresponding to a curtailment rate of {_fmt_num(cur_rate, 4)}%."
        )
        sentences.append(_pick_static([
            "The curtailment rate is a more meaningful indicator than the absolute curtailed volume because it "
            "accounts for project scale, making comparisons across different sites more reliable.",
            "Curtailment rate normalises the loss against project output, so it is the better figure to compare across sites of different sizes.",
            "The rate figure is more comparable across sites than the raw MWh number, since it adjusts for project scale.",
            "Looking at the rate rather than the raw volume gives a cleaner picture when comparing this site against others.",
            "The curtailment rate is the key figure for cross-site comparison — it removes the effect of differing project scales.",
        ], seed_val=curtailment))
    elif curtailment is not None:
        sentences.append(
            f"Based on the zone substation load and grid import/export headroom data for this area, "
            f"the inferred curtailment during the assessment period is approximately {_fmt_num(curtailment, 2)} MWh."
        )
    elif cur_rate is not None:
        sentences.append(
            f"Based on the zone substation load and grid import/export headroom data for this area, "
            f"the inferred curtailment rate is {_fmt_num(cur_rate, 4)}%."
        )
    return " ".join(sentences)


def _opening(row: Dict[str, Any], rng: Any, style: str = "normal") -> str:
    pfi = _safe_get(row, "pfi")
    prob = _safe_get(row, "risk_probability", 0.0)
    level = risk_level_from_row(row)
    pct = _fmt_pct(prob)
    frame = _risk_frame(level)
    score = _risk_score_str(row) or "N/A"
    options = {
        "normal": NORMAL_OPENING_TEMPLATES,
        "reason": REASON_OPENING_TEMPLATES,
        "short": SHORT_OPENING_TEMPLATES,
    }
    return _pick(rng, options.get(style, NORMAL_OPENING_TEMPLATES)).format(
        pfi=pfi,
        level=level,
        pct=pct,
        frame=frame,
        score=score,
    )


def generate_normal_answer(row: Dict[str, Any], rng: Any) -> Tuple[str, Dict[str, bool]]:
    level = risk_level_from_row(row)
    direction = _signal_direction(row)
    mostly_low_mh = level in {"Medium", "High"} and direction == "mostly_low"
    parts = [_opening(row, rng, "normal")]
    bridge = _bridge_explanation(row, rng)
    if bridge:
        parts.append(bridge)
    else:
        drivers = _sample_drivers(row, rng, limit=3, include_low_support=(level == "Low"))
        parts.append(_join_sentence(drivers))
    inferred = _inferred_driver_note(row, rng)
    if inferred:
        parts.append(inferred)
    grid = _grid_context(row)
    if grid and int(rng.integers(0, 2)) == 1:
        parts.append(grid)
    return " ".join(p for p in parts if p), _gen_stats(bool(bridge), mostly_low_mh)


def generate_reason_answer(row: Dict[str, Any], rng: Any) -> Tuple[str, Dict[str, bool]]:
    level = risk_level_from_row(row)
    direction = _signal_direction(row)
    mostly_low_mh = level in {"Medium", "High"} and direction == "mostly_low"
    parts = [_opening(row, rng, "reason")]
    bridge = _bridge_explanation(row, rng)
    if bridge:
        parts.append(bridge)
    drivers = _sample_drivers(row, rng, limit=4, include_low_support=(level == "Low"))
    if drivers:
        parts.append("The main evidence in the retrieved features is: " + _join_sentence(drivers).rstrip(".") + ".")
    inferred = _inferred_driver_note(row, rng)
    if inferred:
        parts.append(inferred)
    grid = _grid_context(row)
    if grid:
        parts.append(grid)
    return "\n\n".join(p for p in parts if p), _gen_stats(bool(bridge), mostly_low_mh)


def generate_short_answer(row: Dict[str, Any], rng: Any) -> Tuple[str, Dict[str, bool]]:
    return _opening(row, rng, "short"), _gen_stats(False, False)


def generate_detailed_answer(row: Dict[str, Any], rng: Any) -> Tuple[str, Dict[str, bool]]:
    level = risk_level_from_row(row)
    direction = _signal_direction(row)
    mostly_low_mh = level in {"Medium", "High"} and direction == "mostly_low"
    parts = [_opening(row, rng, "normal")]

    # Emergency response
    fac5 = _safe_get(row, "total_facilities_5km")
    nearest = _safe_get(row, "closest_facility_distance")
    if fac5 is not None or nearest is not None:
        fac_count = int(float(fac5)) if fac5 is not None else None
        dist = float(nearest) if nearest is not None else None
        resp_parts: List[str] = []
        if fac_count is not None:
            noun = "facility" if fac_count == 1 else "facilities"
            resp_parts.append(
                f"There {'is' if fac_count == 1 else 'are'} {fac_count} fire-response {noun} within 5 km of this site"
            )
        if dist is not None:
            tail = f"with the nearest about {dist:.0f} m away" if resp_parts else f"The nearest fire-response facility is about {dist:.0f} m away"
            resp_parts.append(tail)
        sentence = (", ".join(resp_parts) if resp_parts else "") + "."
        if dist is not None and dist > 4000:
            sentence += " This distance may constrain emergency response time."
        parts.append(sentence)

    # Vegetation and fuel context — natural sentences
    veg_parts: List[str] = []
    is_prone_v = _to_float(_safe_get(row, "is_prone"))
    veg_v      = _to_float(_safe_get(row, "veg_area"))
    evc_bcs_v  = _to_float(_safe_get(row, "evc_bcs_0"))
    evc_mut_v  = _to_float(_safe_get(row, "evc_mut_1"))
    xg2_v      = _to_float(_safe_get(row, "xgroup_2"))
    xg7_v      = _to_float(_safe_get(row, "xgroup_7"))
    if is_prone_v is not None:
        if is_prone_v >= 0.5:
            veg_parts.append("This property is located within a mapped bushfire-prone area, a significant prior risk indicator.")
        else:
            veg_parts.append("This property does not fall within a mapped bushfire-prone area.")
    if veg_v is not None:
        if veg_v > 1:
            veg_parts.append(f"The native vegetation cover indicator for this parcel is approximately {veg_v:g} sq m, signalling a substantial fuel load in the vicinity and an elevated risk of fire spread.")
        elif veg_v == 0:
            veg_parts.append("The native vegetation cover indicator for this parcel is 0 sq m, meaning there is effectively no native vegetation cover recorded here. This suggests a relatively low surface fuel load and limited fire-spread risk from vegetation.")
        else:
            veg_parts.append(f"The native vegetation cover indicator for this parcel is {veg_v:g} sq m, suggesting a relatively modest fuel load and lower fire-spread risk compared to heavily vegetated sites.")
    if evc_bcs_v is not None and evc_bcs_v >= 0.5:
        veg_parts.append("The vegetation at this site carries an EVC endangered classification, which often implies a complex and dense fuel structure.")
    elif evc_mut_v is not None and evc_mut_v >= 0.5:
        veg_parts.append("The mosaic vegetation structure may create natural fuel discontinuities that could slow fire spread.")
    if xg2_v is not None and xg2_v >= 0.5:
        veg_parts.append("The site falls within a Lower Slopes or Hills Woodlands vegetation class, where slope and wind exposure can amplify fire intensity.")
    if xg7_v is not None and xg7_v >= 0.5:
        veg_parts.append("The riparian or swampy scrub woodland class can buffer fire risk under normal conditions but may become a significant fuel source during drought periods.")
    if veg_parts:
        parts.append(" ".join(veg_parts))

    # Historical fire activity — natural sentences
    fire_parts: List[str] = []
    fc_v  = _to_float(_safe_get(row, "fire_count"))
    yrs_v = _to_float(_safe_get(row, "yrs_since_last_burn"))
    if fc_v is not None:
        # Skip the zero-count note when the prolonged-gap sentence already covers it
        if fc_v == 0 and (yrs_v is None or yrs_v < 20):
            fire_parts.append("There are no historical fire records for this area.")
        elif fc_v == 1:
            fire_parts.append("One fire event has been recorded in this area's history.")
        elif fc_v is not None and fc_v > 1:
            fire_parts.append(f"This area has {int(fc_v)} recorded fire events in its history.")
    if yrs_v is not None:
        if yrs_v >= 20:
            fire_parts.append(
                f"There have been no fire records for this area in the past {yrs_v:g} years. "
                "Such a prolonged interval without burning typically means that surface fuel loads have been accumulating over time, which can significantly increase fire intensity if ignition does occur."
            )
        elif yrs_v < 5:
            fire_parts.append(f"A burn occurred only {yrs_v:g} years ago, which may have reduced the current fuel load.")
        else:
            fire_parts.append(f"The last recorded burn was {yrs_v:g} years ago.")
    if fire_parts:
        parts.append(" ".join(fire_parts))

    inferred = _inferred_driver_note(row, rng)
    if inferred:
        parts.append(inferred)

    grid = _grid_context(row)
    if grid:
        parts.append(grid)

    bridge = _bridge_explanation(row, rng)
    if bridge:
        parts.append(bridge)
    return "\n\n".join(p for p in parts if p), _gen_stats(bool(bridge), mostly_low_mh)


def generate_not_found_answer(fake_pfi: int, rng: Any) -> Tuple[str, Dict[str, bool]]:
    options = [
        f"I could not find a retrieved property record for PFI {fake_pfi}. Please check the identifier or provide another PFI so the risk database can be queried correctly.",
        f"No matching record was retrieved for PFI {fake_pfi}. I should not invent a risk level; please verify the PFI or provide a different property reference.",
        f"PFI {fake_pfi} is not present in the retrieved data. A property-specific risk explanation requires a valid record from the database.",
        f"The lookup returned no record for PFI {fake_pfi}. Please provide a valid PFI, postcode, suburb, or coordinates for a new lookup.",
        f"I cannot see PFI {fake_pfi} in the retrieved records, so I cannot give a reliable risk level for it.",
        f"PFI {fake_pfi} did not match any available property record. Could you double-check the number and try again?",
        f"There is not enough retrieved data for PFI {fake_pfi} to produce a site-specific assessment.",
        f"I do not want to guess here: PFI {fake_pfi} was not found in the lookup results.",
        f"The property identifier {fake_pfi} does not appear to be available in the current data extract.",
        f"I could not retrieve a risk record for PFI {fake_pfi}. A valid property record is needed before I can interpret risk factors.",
    ]
    return _pick(rng, options), _gen_stats(False, False)


def generate_no_pfi_answer(_user_question: str, rng: Any) -> Tuple[str, Dict[str, bool]]:
    options = [
        "I need a property reference before I can retrieve a site-specific risk result. Please provide a PFI, postcode, suburb, or coordinates.",
        "I cannot give a property-specific bushfire or curtailment-risk explanation without a lookup key. Could you share the PFI or another location reference?",
        "To answer accurately, I need the property identifier used by the retrieval database, such as a PFI. Without it, I would be guessing.",
        "Please provide a PFI or location reference first; the model is designed to explain retrieved structured records, not infer a property from an unspecified request.",
        "I can help, but I need to know which property to look up. A PFI is best, but a postcode, suburb, or coordinates can also help.",
        "There is no property-specific record attached to this question yet. Please send a PFI or another location reference.",
        "I should not assign a risk level without a property identifier. Could you provide the PFI for the site?",
        "Before I can assess the risk, I need a lookup value such as a PFI, postcode, suburb, or latitude/longitude.",
        "Please share the property reference first. Once I have the retrieved record, I can explain the risk level and drivers.",
        "I do not have enough information to identify the site. Send the PFI or location details and I can check the risk record.",
    ]
    return _pick(rng, options), _gen_stats(False, False)


_ADVICE_BY_LEVEL = {
    "Low": [
        "keep the site on routine monitoring rather than treating it as a binding exclusion",
        "maintain basic vegetation and asset-management practices before the fire season",
        "confirm that the low result remains valid if project capacity or grid-connection assumptions change",
        "continue checking official fire danger warnings during high-risk weather",
        "keep the risk record updated if new fire-history or vegetation data becomes available",
        "treat the site as relatively favourable, while still checking local planning and access constraints",
        "review curtailment assumptions if the proposed project capacity changes",
        "keep emergency access and vegetation management in the normal due-diligence checklist",
        "avoid assuming zero risk; use the low band as a screening signal rather than a safety guarantee",
        "compare the site with nearby alternatives if grid capacity or land-use constraints become tighter",
    ],
    "Medium": [
        "treat the parcel as a due-diligence item before committing to development",
        "review vegetation/fuel exposure and emergency-response access in more detail",
        "check whether project sizing or battery co-location could reduce curtailment pressure",
        "prepare a clear bushfire management and emergency-response plan before investment decisions",
        "look more closely at the feature values that are pushing the site above a low-risk profile",
        "consider whether nearby fire-response access is adequate for the intended asset type",
        "test sensitivity to project capacity, because curtailment pressure can change with scale",
        "ask for an engineering review if both fire exposure and grid constraints are material",
        "use this result as a watch-list flag rather than an automatic rejection",
        "compare the site against lower-risk parcels before making a final siting decision",
    ],
    "High": [
        "treat the parcel as a major screening constraint rather than a routine candidate",
        "conduct detailed bushfire engineering and grid-connection review before progressing",
        "consider alternative nearby sites or lower-capacity configurations if curtailment or fire exposure is material",
        "use a conservative leave-early and asset-protection strategy for any operational planning",
        "do not rely on the headline risk level alone; review the fire, vegetation, access, and grid indicators together",
        "treat further development as conditional on stronger mitigation evidence",
        "check whether the project can be redesigned, down-sized, or paired with storage to reduce operational pressure",
        "prioritise independent validation of bushfire exposure before any investment commitment",
        "compare this parcel with lower-risk candidates before proceeding",
        "document the risk drivers clearly so the decision can be reviewed by planners, engineers, and emergency specialists",
    ],
}


def generate_advice_answer(row: Dict[str, Any], rng: Any) -> Tuple[str, Dict[str, bool]]:
    level = risk_level_from_row(row)
    direction = _signal_direction(row)
    mostly_low_mh = level in {"Medium", "High"} and direction == "mostly_low"
    bridge = _bridge_explanation(row, rng)
    intro = _opening(row, rng, "normal")
    if bridge:
        intro += " " + bridge
    advice = list(_ADVICE_BY_LEVEL.get(level, _ADVICE_BY_LEVEL["Medium"]))
    rng.shuffle(advice)
    bullets = "\n".join(f"- {item[0].upper() + item[1:]}." for item in advice[:3])
    closing = (
        "These are planning and preparedness recommendations, not a guarantee of safety or a substitute "
        "for advice from fire authorities, grid operators, or project engineers."
    )
    return f"{intro} Recommended next steps:\n{bullets}\n\n{closing}", _gen_stats(bool(bridge), mostly_low_mh)


# ---------------------------------------------------------------------------
# Curtailment / grid constraint focus
# ---------------------------------------------------------------------------

_CURTAILMENT_OPENER = [
    "Reviewing the grid and curtailment profile for PFI {pfi} from the retrieved data:",
    "For PFI {pfi}, the retrieved renewable-project and grid-constraint data show the following:",
    "The curtailment and generation-constraint picture for PFI {pfi}, drawn from the retrieved values:",
    "Examining the grid context for PFI {pfi} using only the retrieved structured data:",
    "Here is the curtailment risk assessment for PFI {pfi} based on the retrieved project data:",
]


def generate_curtailment_answer(row: Dict[str, Any], rng: Any) -> Tuple[str, Dict[str, bool]]:
    pfi = _safe_get(row, "pfi")
    level = risk_level_from_row(row)
    prob = _safe_get(row, "risk_probability", 0.0)
    pct = _fmt_pct(prob)
    capacity = _to_float(_safe_get(row, "P_project"))
    curtailment_vol = _to_float(_safe_get(row, "curtailment"))
    cur_rate = _to_float(_safe_get(row, "cur_rate"))

    score = _risk_score_str(row) or "N/A"
    if capacity is None and curtailment_vol is None and cur_rate is None:
        text = (
            f"The retrieved record for PFI {pfi} does not include project capacity or curtailment data. "
            f"The overall site-screening result is {level} risk ({pct}; risk score {score}), but grid constraint risk cannot "
            f"be assessed further from the current retrieved record."
        )
        return text, _gen_stats(False, False)

    parts = [_pick(rng, _CURTAILMENT_OPENER).format(pfi=pfi)]

    items: List[str] = []
    if capacity is not None:
        items.append(f"Recommended project capacity for this site: {capacity:.2f} MW")
    if curtailment_vol is not None:
        items.append(f"Curtailment inferred from zone substation load and import/export headroom: {curtailment_vol:.2f} MWh")
    if cur_rate is not None:
        if cur_rate >= 0.10:
            severity, pressure = "elevated", "material grid-export constraint pressure"
        elif cur_rate >= 0.05:
            severity, pressure = "moderate", "moderate grid-export constraint pressure"
        else:
            severity, pressure = "low", "limited current grid-export constraint pressure"
        items.append(f"Curtailment rate: {cur_rate:.4f}% — {severity} ({pressure})")
    if items:
        parts.append("\n".join(f"- {item}" for item in items))

    if cur_rate is not None:
        if cur_rate >= 0.10:
            interp = _pick(rng, [
                "A curtailment rate at this level is a material input for any project proforma: it reduces effective generation revenue and signals that the local connection point may be operating near its thermal limit.",
                "This curtailment rate warrants careful review of connection and dispatch assumptions before committing to a project at this location.",
                "Elevated curtailment rates like this indicate that the grid may not be able to absorb all generation at this node, which directly affects project revenue and IRR.",
            ])
        elif cur_rate < 0.03:
            interp = _pick(rng, [
                "The low curtailment rate suggests limited grid-export constraint at this location, which is relatively favourable for renewable project revenue certainty.",
                "A curtailment rate at this level does not raise significant grid-constraint concerns in the retrieved assessment.",
                "This low curtailment rate is a positive signal for generation dispatch reliability at this site.",
            ])
        else:
            interp = _pick(rng, [
                "The curtailment rate sits in a moderate range and should be included in project financial modelling.",
                "This is a mid-range curtailment result — not a showstopper, but worth carrying into the revenue model.",
                "A curtailment rate at this level is a planning input rather than a hard constraint; include it in the project proforma.",
                "The retrieved curtailment rate is moderate; it does not disqualify the site outright but should inform dispatch and revenue assumptions.",
                "This curtailment level is not immediately disqualifying, though it may become more significant as additional capacity is added to the same network zone.",
                "The grid shows moderate constraint pressure at this location — factor the curtailment rate into financial modelling before committing to the project.",
            ])
        parts.append(interp)

    link = _pick(rng, [
        f"Taken together with the bushfire-risk indicators, the combined site-screening result for PFI {pfi} is {level} ({pct}; risk score {score}).",
        f"The full dual-risk assessment places PFI {pfi} in the {level} band, with the curtailment profile contributing to that overall classification ({pct}; risk score {score}).",
        f"The overall screening classification for PFI {pfi} is {level} risk ({pct}; risk score {score}), with grid constraint being one of the contributing risk dimensions.",
    ])
    parts.append(link)
    return "\n\n".join(p for p in parts if p), _gen_stats(False, False)


# ---------------------------------------------------------------------------
# Site selection / siting decision
# ---------------------------------------------------------------------------

_SITING_OPENER = {
    "Low": [
        "Based on the retrieved dual-risk assessment, PFI {pfi} is a relatively favourable candidate for renewable energy siting at this screening stage (retrieved probability {pct}; risk score {score}).",
        "PFI {pfi} returns a Low risk classification in the combined site-screening model ({pct}; risk score {score}), placing it in a comparatively viable position for further development assessment.",
        "The retrieved screening result for PFI {pfi} is Low risk ({pct}; risk score {score}). At this stage, the site does not raise major screening constraints.",
    ],
    "Medium": [
        "PFI {pfi} comes back as Medium risk in the retrieved dual-risk assessment ({pct}; risk score {score}). Further due diligence is required before committing to development.",
        "Based on the retrieved data, PFI {pfi} sits in the Medium risk band ({pct}; risk score {score}). It is not an automatic rejection but warrants careful review of both fire exposure and grid constraints.",
        "The retrieved assessment classifies PFI {pfi} as Medium risk ({pct}; risk score {score}). This is a watch-list result that should not be treated as low concern.",
    ],
    "High": [
        "The retrieved dual-risk assessment classifies PFI {pfi} as High risk ({pct}; risk score {score}). This is a significant screening constraint that should be addressed before progressing.",
        "PFI {pfi} returns a High risk classification ({pct}; risk score {score}) in the combined site-screening model. I would treat this as a material barrier to development without further mitigation evidence.",
        "Based on the retrieved data, PFI {pfi} is a high-constraint candidate ({pct}; risk score {score}) for renewable energy siting — detailed engineering and risk review is required before investment commitment.",
    ],
}


def generate_siting_answer(row: Dict[str, Any], rng: Any) -> Tuple[str, Dict[str, bool]]:
    pfi = _safe_get(row, "pfi")
    level = risk_level_from_row(row)
    prob = _safe_get(row, "risk_probability", 0.0)
    pct = _fmt_pct(prob)
    mostly_low_mh = level in {"Medium", "High"} and _signal_direction(row) == "mostly_low"

    score = _risk_score_str(row) or "N/A"
    opening = _pick(rng, _SITING_OPENER.get(level, _SITING_OPENER["Medium"])).format(pfi=pfi, pct=pct, score=score)

    fire_items: List[str] = []
    is_prone = _to_float(_safe_get(row, "is_prone"))
    veg = _to_float(_safe_get(row, "veg_area"))
    fire_count = _to_float(_safe_get(row, "fire_count"))
    yrs = _to_float(_safe_get(row, "yrs_since_last_burn"))
    nearest = _to_float(_safe_get(row, "closest_facility_distance"))
    fac5 = _to_float(_safe_get(row, "total_facilities_5km"))

    if is_prone is not None:
        fire_items.append(f"Bushfire-prone area overlay: {_yes_no(is_prone)}")
    if veg is not None:
        note = "notable fuel exposure" if veg > 1 else "low fuel indicator"
        fire_items.append(f"Native vegetation cover: {veg:g} sq m — {note}")
    if fire_count is not None:
        fire_items.append(f"Historical fire count: {fire_count:g}")
    if yrs is not None:
        if yrs >= 20:
            fuel_note = "accumulated fuel load likely"
        elif yrs < 5:
            fuel_note = "recent burn may have reduced fuel load"
        else:
            fuel_note = "moderate fuel accumulation possible"
        fire_items.append(f"Years since last burn: {yrs:g} ({fuel_note})")
    if nearest is not None:
        access = "response access constrained" if nearest > 4000 else "reasonable emergency response access"
        fire_items.append(f"Nearest fire-response facility: {nearest:.0f} m ({access})")
    elif fac5 is not None:
        fire_items.append(f"Fire-response facilities within 5 km: {int(fac5)}")

    grid_items: List[str] = []
    capacity = _to_float(_safe_get(row, "P_project"))
    curtailment_vol = _to_float(_safe_get(row, "curtailment"))
    cur_rate = _to_float(_safe_get(row, "cur_rate"))

    if capacity is not None:
        grid_items.append(f"Recommended project capacity: {capacity:.2f} MW")
    if curtailment_vol is not None:
        grid_items.append(f"Curtailment inferred from zone substation load and import/export headroom: {curtailment_vol:.2f} MWh")
    if cur_rate is not None:
        if cur_rate >= 0.10:
            constraint_note = "elevated grid-export constraint — material project risk"
        elif cur_rate >= 0.05:
            constraint_note = "moderate grid constraint — include in financial modelling"
        else:
            constraint_note = "low current curtailment pressure"
        grid_items.append(f"Curtailment rate: {cur_rate:.4f}% ({constraint_note})")

    lines = [opening, ""]
    if fire_items:
        lines.append("Fire and physical risk indicators:")
        lines.extend(f"  - {item}" for item in fire_items)
    if grid_items:
        lines.append("")
        lines.append("Grid and curtailment risk indicators:")
        lines.extend(f"  - {item}" for item in grid_items)

    advice = list(_ADVICE_BY_LEVEL.get(level, _ADVICE_BY_LEVEL["Medium"]))
    rng.shuffle(advice)
    lines += ["", "Recommended next steps:"]
    lines.extend(f"  - {a[0].upper() + a[1:]}." for a in advice[:2])

    bridge = _bridge_explanation(row, rng)
    if bridge:
        lines += ["", bridge]

    lines += [
        "",
        "This assessment is a first-pass site screening based on the retrieved model outputs. "
        "It does not substitute for detailed engineering review, fire authority consultation, or a formal grid connection study.",
    ]
    return "\n".join(lines), _gen_stats(bool(bridge), mostly_low_mh)


# ---------------------------------------------------------------------------
# Dual-risk combined assessment
# ---------------------------------------------------------------------------

def generate_dual_risk_answer(row: Dict[str, Any], rng: Any) -> Tuple[str, Dict[str, bool]]:
    pfi = _safe_get(row, "pfi")
    level = risk_level_from_row(row)
    prob = _safe_get(row, "risk_probability", 0.0)
    pct = _fmt_pct(prob)
    mostly_low_mh = level in {"Medium", "High"} and _signal_direction(row) == "mostly_low"

    score = _risk_score_str(row) or "N/A"
    intro = _pick(rng, [
        f"The retrieved dual-risk assessment for PFI {pfi} covers two dimensions: fire exposure and grid/curtailment constraints. Combined screening result: {level} ({pct}; risk score {score}).",
        f"PFI {pfi} is assessed across two risk domains in the retrieved screening model. The combined classification is {level} risk at {pct} (risk score {score}).",
        f"For PFI {pfi}, the site-screening model integrates bushfire indicators with grid and curtailment data. The retrieved combined result is {level} risk ({pct}; risk score {score}).",
    ])

    # --- Fire risk section ---
    fire_lines = [f"Bushfire Risk Analysis ({level}, {pct}, risk score {score})"]
    fire_lines.append(_opening(row, rng, "normal"))
    fire_drivers = _feature_driver_phrases(row, include_low_support=(level == "Low"))
    if fire_drivers:
        fire_lines.append("Key fire risk indicators:")
        fire_lines.extend(f"  - {d[0].upper() + d[1:]}." for d in fire_drivers[:4])
    bridge = _bridge_explanation(row, rng)
    if bridge:
        fire_lines.append(bridge)

    # --- Grid / curtailment section ---
    capacity = _to_float(_safe_get(row, "P_project"))
    curtailment_vol = _to_float(_safe_get(row, "curtailment"))
    cur_rate = _to_float(_safe_get(row, "cur_rate"))

    grid_lines = ["Grid and Curtailment Risk Analysis"]
    if capacity is None and curtailment_vol is None and cur_rate is None:
        grid_lines.append(
            "No project capacity or curtailment data is present in the retrieved record. "
            "Grid constraint risk cannot be assessed from the current data extract."
        )
    else:
        grid_items: List[str] = []
        if capacity is not None:
            grid_items.append(f"Recommended project capacity: {capacity:.2f} MW")
        if curtailment_vol is not None:
            grid_items.append(f"Curtailment inferred from zone substation load and import/export headroom: {curtailment_vol:.2f} MWh")
        if cur_rate is not None:
            grid_items.append(f"Curtailment rate: {cur_rate:.4f}%")
        if grid_items:
            grid_lines.extend(f"  - {item}" for item in grid_items)
        if cur_rate is not None and cur_rate >= 0.10:
            grid_lines.append(
                "The retrieved curtailment rate exceeds 10%, indicating material grid-export constraint pressure. "
                "This is a significant input for project revenue modelling and connection feasibility assessment."
            )
        elif cur_rate is not None and cur_rate < 0.03:
            grid_lines.append(
                "The retrieved curtailment rate is low, suggesting limited grid-export constraint at this location. "
                "This is a favourable signal for renewable project dispatch reliability."
            )
        elif capacity is not None:
            grid_lines.append(
                "The curtailment rate is in a moderate range. It should be included in project financial modelling but is not immediately disqualifying."
            )

    parts = [intro, "\n".join(fire_lines), "\n".join(grid_lines)]
    return "\n\n".join(p for p in parts if p), _gen_stats(bool(bridge), mostly_low_mh)


def generate_simple_answer(row: Dict[str, Any], rng: Any) -> Tuple[str, Dict[str, bool]]:
    pfi = _safe_get(row, "pfi")
    level = risk_level_from_row(row)
    pct = _fmt_pct(_safe_get(row, "risk_probability", 0.0))
    plain = {
        "Low": [
            f"Simply put, PFI {pfi} looks lower risk in the retrieved assessment. The probability is about {pct}, so this site is not currently flagged as a major concern, but normal bushfire awareness still matters.",
            f"Plain English: PFI {pfi} is in the Low risk band. That means the retrieved data does not show strong warning signs compared with higher-risk sites.",
            f"PFI {pfi} looks like a lower-risk option. I would still keep normal checks in place, but the retrieved result is not raising a major red flag.",
            f"In everyday terms, PFI {pfi} is on the lower end of the risk scale, with a retrieved probability of {pct}.",
            f"The simple takeaway is that PFI {pfi} is currently classed as Low risk, so it is less concerning than medium or high sites.",
            f"PFI {pfi} is not risk-free, but the retrieved result puts it in the Low band.",
            f"For a non-technical reader: this site looks comparatively favourable in the retrieved risk data.",
            f"PFI {pfi} has a Low label, meaning the available record does not show strong risk pressure.",
            f"The plain answer is: PFI {pfi} is a lower-risk site, based on the retrieved data.",
            f"PFI {pfi} appears relatively manageable from the retrieved risk result, but it should still be monitored like any site.",
        ],
        "Medium": [
            f"Simply put, PFI {pfi} is not the lowest-risk option, but it is also not the most severe category. The retrieved probability is about {pct}, so it deserves further checking before a site decision.",
            f"Plain English: PFI {pfi} sits in the Medium band. Some risk factors or model evidence are enough to keep it on the watch list.",
            f"PFI {pfi} is a middle-ground case. It is not an automatic rejection, but it should not be treated as low concern either.",
            f"In simple terms, PFI {pfi} needs a closer look. The retrieved result is Medium, with probability around {pct}.",
            f"The simple takeaway is that PFI {pfi} has enough risk evidence to deserve due diligence.",
            f"PFI {pfi} is not in the highest band, but the retrieved result suggests more checking is needed.",
            f"For a non-technical reader: this site is on the watch list, not the clear low-risk list.",
            f"PFI {pfi} has a Medium label, so I would treat it as manageable only after reviewing the drivers.",
            f"The plain answer is: PFI {pfi} has moderate screening risk and should be checked before moving forward.",
            f"PFI {pfi} is a caution case. The result is not extreme, but it is not something to ignore.",
        ],
        "High": [
            f"Simply put, PFI {pfi} is a high-risk site in the retrieved assessment. The probability is about {pct}, so it should be treated cautiously for renewable project planning.",
            f"Plain English: PFI {pfi} is in the High risk band. That does not describe a certain event, but it does mean the site has elevated screening risk.",
            f"PFI {pfi} should be treated carefully. The retrieved result puts it in the High band, so it is a serious due-diligence item.",
            f"In simple terms, PFI {pfi} raises a strong warning flag in the retrieved risk data.",
            f"The simple takeaway is that PFI {pfi} is not a routine low-risk candidate.",
            f"PFI {pfi} has a High label, so I would not progress it without stronger checks and mitigation planning.",
            f"For a non-technical reader: this site has elevated risk and needs careful review before any decision.",
            f"PFI {pfi} is classed as High risk, which means the site should be handled conservatively.",
            f"The plain answer is: PFI {pfi} is a high-risk screening result, not just a minor concern.",
            f"PFI {pfi} looks like a constrained site. The retrieved result suggests extra caution is needed.",
        ],
    }
    text = _pick(rng, plain.get(level, plain["Medium"]))
    bridge = _bridge_explanation(row, rng)
    if bridge:
        text += " " + bridge
    grid = _grid_context(row)
    if grid:
        text += "\n\n" + grid
    return text, _gen_stats(bool(bridge), level in {"Medium", "High"} and _signal_direction(row) == "mostly_low")


# ---------------------------------------------------------------------------
# Methodology / accuracy inquiry
# ---------------------------------------------------------------------------

def generate_methodology_answer(row: Dict[str, Any], rng: Any) -> Tuple[str, Dict[str, bool]]:
    """
    Answers questions like 'How do you know this is correct?' or
    'Where does this data come from?' by explaining the ML pipeline and data sources.
    If a row is provided, references the specific retrieved values as an example.
    """
    pfi   = _safe_get(row, "pfi")
    level = risk_level_from_row(row)
    prob  = _safe_get(row, "risk_probability", 0.0)
    pct   = _fmt_pct(prob)
    score = _risk_score_str(row) or "N/A"

    intro = _pick(rng, [
        f"The {level} risk classification for PFI {pfi} (probability {pct}, risk score {score}) is not a guess — it is the output of a machine learning model trained on real geospatial and grid data for Victoria, Australia.",
        f"The retrieved result for PFI {pfi} — {level} risk at {pct} (risk score {score}) — comes from a supervised machine learning pipeline, not from a rule-based lookup or manual judgement.",
        f"This risk result ({level}, {pct}, risk score {score}) is produced by a trained model, not inferred from memory or general knowledge about the area.",
    ])

    source_para = _pick(rng, [
        (
            "The model is trained on several data sources: "
            "mapped bushfire-prone area overlays and ecological vegetation class (EVC) data from state spatial datasets; "
            "historical fire event records including cumulative fire count and years since last burn; "
            "emergency response facility locations used to compute the number of fire-response facilities within 5 km and the distance to the nearest one; "
            "zone substation load profiles and import/export capacity headroom from distribution network annual planning reports, "
            "which are the basis for the inferred curtailment and curtailment rate; "
            "and renewable project records used to calibrate the recommended installed capacity at each location."
        ),
        (
            "The underlying data covers multiple dimensions of site risk. "
            "Fire and vegetation data include bushfire-prone area status, ecological vegetation class indicators, "
            "cumulative fire count, and years since the last recorded burn — "
            "all sourced from Victorian government spatial datasets. "
            "Emergency access is measured using the number and proximity of fire-response facilities within a 5 km radius. "
            "Grid constraint data comes from publicly available Distribution Annual Planning Reports from zone substation operators, "
            "which provide load profiles, import and export capacity limits, and actual curtailment volumes."
        ),
    ])

    model_para = _pick(rng, [
        (
            "A LightGBM classifier is trained using a semi-supervised self-training approach. "
            "High-confidence anchor samples — existing renewable project parcels as low-risk examples, "
            "and high fire-count bushfire-prone parcels as high-risk examples — seed the initial training. "
            "The model is then applied to unlabelled parcels to generate pseudo-labels for the highest-confidence predictions, "
            "and the process iterates. "
            "The final output is a calibrated risk probability and a composite risk score, "
            "which are stored in a database and retrieved at query time rather than being predicted by the language model from memory."
        ),
        (
            "The risk probability and score are produced by a gradient-boosted decision tree model (LightGBM) "
            "trained in a semi-supervised framework on Victoria parcel data. "
            "The model learns from labelled anchor samples and expands through iterative pseudo-labelling of unlabelled parcels. "
            "All results are pre-computed and stored in a database indexed by PFI. "
            "When you ask about a property, the system retrieves the stored record and this language model explains it — "
            "it does not re-predict the risk from the features you can see in the input."
        ),
    ])

    caveat = _pick(rng, [
        "As with any screening model, the result reflects the data available at the time of assessment. "
        "It is a first-pass site-screening tool and is not a substitute for detailed engineering review, "
        "fire authority consultation, or a formal grid connection study.",
        "The model provides a data-driven screening result, but it should be treated as a starting point rather than a definitive conclusion. "
        "Independent verification by fire safety engineers, grid connection specialists, and planners is recommended before any investment decision.",
        "This is a systematic screening tool, not a guarantee of safety or revenue certainty. "
        "The risk result should inform — not replace — professional technical review.",
    ])

    return "\n\n".join([intro, source_para, model_para, caveat]), _gen_stats(False, False)


# ---------------------------------------------------------------------------
# Closing / farewell
# ---------------------------------------------------------------------------

def generate_closing_answer(rng: Any) -> Tuple[str, Dict[str, bool]]:
    """Responds to thank-you and end-of-conversation messages."""
    options = [
        "You're welcome. Feel free to return whenever you need to screen another property — just send the PFI and I can pull up the risk record.",
        "Happy to help. If you need to assess additional sites or want a different angle on the same property, I'm here.",
        "Glad I could assist. Good luck with the project — if anything else comes up during due diligence, feel free to ask.",
        "No problem at all. Reach out any time if you need further risk assessments or want to explore curtailment figures for other locations.",
        "You're welcome! If new sites come into scope or you need to revisit this one, just send the PFI.",
        "Anytime. Risk screening is an iterative process — come back if the project scope changes or if you want to compare this site against alternatives.",
        "Glad to be of help. If the development progresses and you need a more detailed breakdown of any specific risk dimension, just ask.",
        "Happy to help with the screening. Best of luck with the renewable energy project.",
        "Of course — that's what I'm here for. Feel free to check in with more properties as the pipeline develops.",
        "You're welcome. If you have other sites in the pipeline, bring them through and I can run the same assessment.",
    ]
    return _pick(rng, options), _gen_stats(False, False)
