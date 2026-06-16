"""Input text rendering helpers for build_sft.py."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


FEATURE_FIELD_NAMES = [
    "total_facilities_5km",
    "closest_facility_distance",
    "is_prone",
    "veg_area",
    "evc_mut_1",
    "evc_bcs_0",
    "xgroup_2",
    "xgroup_7",
    "fire_count",
    "yrs_since_last_burn",
    "P_project",
    "curtailment",
    "cur_rate",
]


TASK_NOTE_TEMPLATES = [
    "Use only the retrieved property data above. Do not invent risk values or unsupported property details.",
    "Answer from the retrieved data only. If a field is not shown, do not make it up.",
    "Treat the retrieved risk level and probability as the source of truth, and ground any explanation in the listed features.",
    "Base the response on the retrieved structured data. Keep any reasoning tied to the provided feature values.",
    "Do not use outside assumptions. Use the retrieved risk result and feature context to answer the user.",
    "Stay faithful to the retrieved property record, including the risk level, probability, and available feature explanations.",
]


NO_DATA_TASK_NOTE_TEMPLATES = [
    "Do not invent any risk values. Ask for a valid property identifier or enough information to perform a lookup.",
    "Because no property record was retrieved, do not provide a property-specific risk estimate.",
    "If the data is missing, explain that the lookup cannot be completed and ask for clarification.",
    "Do not guess the property's risk. Request a valid PFI, postcode, suburb, or coordinates.",
]


INPUT_LAYOUT_TEMPLATES = [
    "User question: {user_question}\n\nRetrieved structured data:\n{retrieved_text}\n\nTask note:\n{task_note}",
    "User request:\n{user_question}\n\nRetrieved property data:\n{retrieved_text}\n\nTask note:\n{task_note}",
    "Question from user:\n{user_question}\n\nData retrieved for this property:\n{retrieved_text}\n\nTask note:\n{task_note}",
    "User question:\n{user_question}\n\nRetrieved bushfire risk record:\n{retrieved_text}\n\nTask note:\n{task_note}",
]


NO_DATA_INPUT_LAYOUT_TEMPLATES = [
    "User question: {user_question}\n\nRetrieved structured data:\nNone\n\nReason: {reason}\n\nTask note:\n{task_note}",
    "User request:\n{user_question}\n\nRetrieved property data:\nNone\n\nReason: {reason}\n\nTask note:\n{task_note}",
    "Question from user:\n{user_question}\n\nRetrieved bushfire risk record:\nNone\n\nReason: {reason}\n\nTask note:\n{task_note}",
]


def _select_variant(values: List[str], key: str) -> str:
    if not values:
        return ""
    idx = sum(ord(ch) for ch in str(key)) % len(values)
    return values[idx]


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


def _format_float(value: Any, digits: int = 2) -> str:
    number = _to_float(value)
    if number is None:
        return str(value)
    return f"{number:.{digits}f}"


def _format_compact(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return str(value)
    return f"{number:g}"


def _format_yes_no(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return str(value)
    return "Yes" if number > 0 else "No"


def _prob_percent(probability: Any) -> str:
    number = _to_float(probability)
    if number is None:
        return "Unknown"
    return f"{number * 100:.2f}%"


def _append_feature(
    lines: List[str],
    missing: List[str],
    row: Any,
    field: str,
    label: str,
    explanation: str,
    formatter: Any = _format_compact,
    show_unknown_fields: bool = False,
) -> None:
    value = _safe_get(row, field)
    if value is None:
        missing.append(field)
        if show_unknown_fields:
            lines.append(f"- {label}: Unknown. {explanation}")
        return
    try:
        rendered = formatter(value)
    except Exception:
        rendered = str(value)
    lines.append(f"- {label}: {rendered}. {explanation}")


def render_retrieved_data(
    row: Optional[Dict[str, Any]],
    show_unknown_fields: bool = False,
) -> Tuple[str, List[str], Dict[str, int]]:
    """Render retrieved property data and all configured feature explanations."""
    if row is None:
        return "None", ["all"], {"locality_lines_removed": 0, "unknown_lines_omitted": 0}

    missing: List[str] = []
    lines: List[str] = []

    pfi = _safe_get(row, "pfi", "Unknown")
    probability = _safe_get(row, "risk_probability")
    risk_level = _safe_get(row, "risk_level")
    if risk_level is None:
        risk_level = "Unknown"
    else:
        risk_level = str(risk_level).strip().capitalize()

    lines.append(f"PFI: {pfi}")
    if probability is None:
        missing.append("risk_probability")
        lines.append("Risk probability: Unknown")
    else:
        lines.append(f"Risk probability: {float(probability):.4f} ({_prob_percent(probability)})")
    risk_score = _safe_get(row, "risk_score")
    if risk_score is not None:
        lines.append(f"Risk score: {float(risk_score):.2f}")
    lines.append(f"Risk level: {risk_level}")
    lines.append("")
    lines.append("Feature context:")

    _append_feature(
        lines,
        missing,
        row,
        "total_facilities_5km",
        "Fire-response facilities within 5 km",
        "More nearby fire-response resources can support faster response and may reduce relative risk.",
        formatter=lambda v: f"{int(round(float(v)))}",
        show_unknown_fields=show_unknown_fields,
    )
    _append_feature(
        lines,
        missing,
        row,
        "closest_facility_distance",
        "Distance to nearest fire-response facility",
        "Longer distance can delay emergency response and push risk upward.",
        formatter=lambda v: f"{float(v):.2f} m",
        show_unknown_fields=show_unknown_fields,
    )
    _append_feature(
        lines,
        missing,
        row,
        "is_prone",
        "Intersects a bushfire-prone area",
        "A value of Yes means the property intersects a mapped bushfire-prone zone.",
        formatter=_format_yes_no,
        show_unknown_fields=show_unknown_fields,
    )
    _append_feature(
        lines,
        missing,
        row,
        "veg_area",
        "Maximum native vegetation cover area",
        "Larger native vegetation area indicates more potential fuel and can increase fire spread capacity.",
        formatter=_format_compact,
        show_unknown_fields=show_unknown_fields,
    )
    _append_feature(
        lines,
        missing,
        row,
        "evc_mut_1",
        "EVC mosaic vegetation structure flag",
        "Mosaic vegetation is more discontinuous, so fuel breaks may slow fire spread compared with continuous vegetation.",
        formatter=_format_compact,
        show_unknown_fields=show_unknown_fields,
    )
    _append_feature(
        lines,
        missing,
        row,
        "evc_bcs_0",
        "EVC bioregional conservation status: Endangered flag",
        "Endangered vegetation can indicate high ecological value and complex fuel structure.",
        formatter=_format_compact,
        show_unknown_fields=show_unknown_fields,
    )
    _append_feature(
        lines,
        missing,
        row,
        "xgroup_2",
        "Macro vegetation group: Lower Slopes or Hills Woodlands",
        "Slope and wind exposure in hill woodland settings can accelerate spread and increase fire intensity.",
        formatter=_format_compact,
        show_unknown_fields=show_unknown_fields,
    )
    _append_feature(
        lines,
        missing,
        row,
        "xgroup_7",
        "Macro vegetation group: Riparian Scrubs or Swampy Scrubs and Woodlands",
        "Moisture can suppress fire under normal conditions, but drought can cause risk to rise sharply in these fuels.",
        formatter=_format_compact,
        show_unknown_fields=show_unknown_fields,
    )
    _append_feature(
        lines,
        missing,
        row,
        "fire_count",
        "Historical cumulative fire count",
        "More recorded fires indicate greater historical fire activity around the property.",
        formatter=_format_compact,
        show_unknown_fields=show_unknown_fields,
    )
    _append_feature(
        lines,
        missing,
        row,
        "yrs_since_last_burn",
        "Years since last burn",
        "A longer time since last burn can allow fuels to accumulate, while a recent burn may indicate reduced fuel load.",
        formatter=_format_compact,
        show_unknown_fields=show_unknown_fields,
    )
    _append_feature(
        lines,
        missing,
        row,
        "P_project",
        "Actual installed project capacity",
        "Baseline generation capacity at the relevant project or substation context, measured in MW.",
        formatter=lambda v: f"{_format_float(v, 2)} MW",
        show_unknown_fields=show_unknown_fields,
    )
    _append_feature(
        lines,
        missing,
        row,
        "curtailment",
        "Actual energy curtailment volume",
        "Energy curtailed at the substation context, measured in MWh.",
        formatter=lambda v: f"{_format_float(v, 2)} MWh",
        show_unknown_fields=show_unknown_fields,
    )
    _append_feature(
        lines,
        missing,
        row,
        "cur_rate",
        "Actual curtailment rate",
        "Curtailment divided by total generation, expressed as a percentage.",
        formatter=lambda v: f"{_format_float(v, 4)}%",
        show_unknown_fields=show_unknown_fields,
    )

    omitted = 0 if show_unknown_fields else sum(1 for field in missing if field in FEATURE_FIELD_NAMES)
    return "\n".join(lines), missing, {
        "locality_lines_removed": 0,
        "unknown_lines_omitted": omitted,
    }


def compose_input(user_question: str, retrieved_text: str, with_task_note: bool = True) -> str:
    if not with_task_note:
        return f"User question: {user_question}\n\nRetrieved structured data:\n{retrieved_text}"
    key = user_question + retrieved_text[:80]
    layout = _select_variant(INPUT_LAYOUT_TEMPLATES, key)
    task_note = _select_variant(TASK_NOTE_TEMPLATES, key)
    return layout.format(
        user_question=user_question,
        retrieved_text=retrieved_text,
        task_note=task_note,
    )


def compose_no_data_input(user_question: str, reason: Optional[str] = None) -> str:
    reason_text = reason or "No matching property data was retrieved."
    key = user_question + reason_text
    layout = _select_variant(NO_DATA_INPUT_LAYOUT_TEMPLATES, key)
    task_note = _select_variant(NO_DATA_TASK_NOTE_TEMPLATES, key)
    return layout.format(
        user_question=user_question,
        reason=reason_text,
        task_note=task_note,
    )
