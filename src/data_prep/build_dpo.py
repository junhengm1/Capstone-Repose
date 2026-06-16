#!/usr/bin/env python3
"""
Build DPO preference data for the renewable-energy dual-risk chatbot.

Each output row uses LlamaFactory ranking format:
  {"instruction": str, "input": str, "chosen": str, "rejected": str}

Design goals:
- Default DPO:SFT size ratio is 1:2, using outputs/sft_data_log.json when available.
- Cover the same broad dialogue surface as SFT: lookup, short answer, detailed drivers,
  why explanation, simple explanation, advice, curtailment, siting, dual-risk,
  methodology, closing, not_found, and no_pfi.
- Rejected answers include both obvious wrong-type failures and harder negatives where
  the writing sounds natural but contains a subtle factual or routing error.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sft_input_text import FEATURE_FIELD_NAMES, compose_input, compose_no_data_input, render_retrieved_data
from sft_instruction_text import (
    ADVICE_QUERY_TEMPLATES,
    CLOSING_QUERY_TEMPLATES,
    CURTAILMENT_QUERY_TEMPLATES,
    DETAIL_QUERY_TEMPLATES,
    DUAL_RISK_QUERY_TEMPLATES,
    INSTRUCTION_ADVICE_POOL,
    INSTRUCTION_CLOSING_POOL,
    INSTRUCTION_CURTAILMENT_POOL,
    INSTRUCTION_DETAIL_POOL,
    INSTRUCTION_DUAL_RISK_POOL,
    INSTRUCTION_METHODOLOGY_POOL,
    INSTRUCTION_NORMAL_POOL,
    INSTRUCTION_NOT_FOUND_POOL,
    INSTRUCTION_NO_PFI_POOL,
    INSTRUCTION_SHORT_POOL,
    INSTRUCTION_SIMPLE_POOL,
    INSTRUCTION_SITING_POOL,
    INSTRUCTION_WHY_POOL,
    METHODOLOGY_QUERY_TEMPLATES,
    NORMAL_QUERY_TEMPLATES,
    NOT_FOUND_QUERY_TEMPLATES,
    NO_PFI_QUERY_TEMPLATES,
    SHORT_QUERY_TEMPLATES,
    SIMPLE_QUERY_TEMPLATES,
    SITING_QUERY_TEMPLATES,
    WHY_QUERY_TEMPLATES,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FINAL_RESULT_PATH = str(DATA_DIR / "final_result.parquet")
SFT_LOG_PATH = str(OUTPUTS_DIR / "sft_data_log.json")
DEFAULT_OUT = str(OUTPUTS_DIR / "dpo_data.jsonl")
DEFAULT_LOG = str(OUTPUTS_DIR / "dpo_data_log.json")

SFT_COUNTS_FALLBACK = {
    "normal_query": 1999,
    "short_answer": 1998,
    "detailed_drivers": 2006,
    "why_explanation": 1000,
    "advice": 498,
    "simple_explanation": 700,
    "curtailment_focus": 603,
    "siting_decision": 501,
    "dual_risk": 499,
    "methodology": 399,
    "not_found": 300,
    "no_pfi": 306,
    "closing": 300,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
        if pd.isna(value):
            return default
    except Exception:
        pass
    return value


def _to_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
        return None if pd.isna(number) else number
    except Exception:
        return None


def _risk_level(row: Any) -> str:
    level = _safe_get(row, "risk_level")
    if level is not None:
        text = str(level).strip().capitalize()
        if text in {"Low", "Medium", "High"}:
            return text
    return "Unknown"


def _fmt_pct(value: Any) -> str:
    number = _to_float(value)
    return "N/A" if number is None else f"{number * 100:.2f}%"


def _fmt_score(row: Any) -> str:
    number = _to_float(_safe_get(row, "risk_score"))
    return "N/A" if number is None else f"{number:.2f}"


def _pick(rng: np.random.Generator, values: List[str]) -> str:
    return values[int(rng.integers(0, len(values)))]


def _fmt_query(template: str, row: Dict[str, Any]) -> str:
    values = {
        "pfi": _safe_get(row, "pfi"),
        "risk_level": _risk_level(row),
    }
    try:
        return template.format(**values)
    except KeyError:
        text = template
        for key, value in values.items():
            text = text.replace("{" + key + "}", str(value))
        return text


def _record(instruction: str, input_block: str, chosen: str, rejected: str) -> Dict[str, str]:
    return {
        "instruction": instruction,
        "input": input_block,
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
    }


def _call_generator(fn: Callable, row: Dict[str, Any], rng: np.random.Generator) -> str:
    text, _stats = fn(row, rng)
    return text


def _opposite_level(level: str, rng: np.random.Generator) -> str:
    choices = [x for x in ("Low", "Medium", "High") if x != level]
    return choices[int(rng.integers(0, len(choices)))]


def _paragraphs(*parts: Optional[str]) -> str:
    return "\n\n".join(str(p).strip() for p in parts if p and str(p).strip())


def _headline(row: Dict[str, Any]) -> str:
    pfi = _safe_get(row, "pfi")
    return (
        f"PFI {pfi} is currently assessed as {_risk_level(row)} risk based on the retrieved dual-risk assessment. "
        f"Its estimated risk probability is about {_fmt_pct(_safe_get(row, 'risk_probability'))}, "
        f"with a composite risk score of {_fmt_score(row)}."
    )


def _fire_driver_sentences(row: Dict[str, Any]) -> List[str]:
    is_prone = _to_float(_safe_get(row, "is_prone"))
    veg = _to_float(_safe_get(row, "veg_area"))
    yrs = _to_float(_safe_get(row, "yrs_since_last_burn"))
    fires = _to_float(_safe_get(row, "fire_count"))
    fac5 = _to_float(_safe_get(row, "total_facilities_5km"))
    nearest = _to_float(_safe_get(row, "closest_facility_distance"))
    evc_end = _to_float(_safe_get(row, "evc_bcs_0"))
    lower_hills = _to_float(_safe_get(row, "evc_mvg_1"))
    riparian = _to_float(_safe_get(row, "evc_mvg_3"))

    drivers: List[str] = []
    if is_prone is not None and is_prone >= 0.5:
        drivers.append("the property overlaps with a mapped bushfire-prone area, which is a strong landscape-level warning sign")
    elif is_prone is not None:
        drivers.append("the property is not mapped inside a bushfire-prone area, which reduces one major source of concern")

    if veg is not None and veg > 1:
        drivers.append(f"native vegetation cover is approximately {veg:g} sq m, so there is visible fuel around the parcel")
    elif veg is not None:
        drivers.append("native vegetation cover is recorded as 0 sq m, suggesting limited surface fuel from native vegetation")

    if evc_end is not None and evc_end >= 0.5:
        drivers.append("the EVC conservation-status flag points to endangered or high-value vegetation, which can imply more complex fuel structure")
    if lower_hills is not None and lower_hills >= 0.5:
        drivers.append("the local vegetation group is associated with lower slopes or hills, where slope and wind exposure can increase fire behaviour")
    if riparian is not None and riparian >= 0.5:
        drivers.append("the riparian or swampy vegetation setting may behave differently under drought conditions")
    if fires is not None and fires > 0:
        drivers.append(f"there are {fires:g} recorded historical fire event(s), showing past fire activity")
    if yrs is not None and yrs >= 20:
        drivers.append(f"the area has not burned for around {yrs:g} years, which can mean fuel has accumulated over time")
    if fac5 is not None and fac5 <= 1:
        drivers.append("there are limited fire-response facilities within 5 km")
    if nearest is not None and nearest > 2500:
        drivers.append(f"the nearest fire-response facility is about {nearest:.0f} m away, which could delay emergency response")
    return drivers


def _fire_context(row: Dict[str, Any], max_items: int = 4) -> str:
    drivers = _fire_driver_sentences(row)
    if not drivers:
        return "The fire-risk result comes from the combined pattern of the retrieved site indicators rather than one single visible field."
    return "The main fire-risk context is that " + "; ".join(drivers[:max_items]) + "."


def _structural_note(row: Dict[str, Any]) -> Optional[str]:
    level = _risk_level(row)
    is_prone = _to_float(_safe_get(row, "is_prone"))
    veg = _to_float(_safe_get(row, "veg_area"))
    fires = _to_float(_safe_get(row, "fire_count"))
    fac5 = _to_float(_safe_get(row, "total_facilities_5km"))
    nearest = _to_float(_safe_get(row, "closest_facility_distance"))

    if level not in {"High", "Medium"}:
        return None
    mild_surface = (veg is None or veg < 1) or (fires is None or fires == 0)
    access_concern = (fac5 is not None and fac5 <= 1) or (nearest is not None and nearest > 2500)
    prone_concern = is_prone is not None and is_prone >= 0.5
    if not mild_surface or not (access_concern or prone_concern):
        return None
    if prone_concern and access_concern:
        return (
            "This is why the result can still be elevated even when vegetation cover or historical fire counts look mild: "
            "the broader bushfire-prone overlay and weaker emergency-response access can outweigh those benign-looking indicators."
        )
    if prone_concern:
        return (
            "This is why the result can still be elevated even when some surface indicators look mild: "
            "the mapped bushfire-prone overlay is a broader landscape constraint, not just a count of vegetation or past fires."
        )
    return (
        "This is why the result can still be elevated even when fuel indicators look mild: "
        "limited or distant fire-response access can increase the consequence side of the screening result."
    )


def _grid_context(row: Dict[str, Any]) -> Optional[str]:
    capacity = _to_float(_safe_get(row, "P_project"))
    curtail = _to_float(_safe_get(row, "curtailment"))
    rate = _to_float(_safe_get(row, "cur_rate"))
    if capacity is None and curtail is None and rate is None:
        return None

    parts: List[str] = []
    if capacity is not None:
        parts.append(f"the recommended project capacity is about {capacity:.2f} MW")
    if curtail is not None and rate is not None:
        parts.append(
            f"based on zone substation load and import/export headroom, estimated curtailed energy is {curtail:.2f} MWh "
            f"with a curtailment rate of {rate:.4f}%"
        )
    elif curtail is not None:
        parts.append(f"estimated curtailed energy is {curtail:.2f} MWh")
    elif rate is not None:
        parts.append(f"the inferred curtailment rate is {rate:.4f}%")
    return "For renewable energy development, " + ", and ".join(parts) + "."


def _curtailment_implication(row: Dict[str, Any]) -> str:
    rate = _to_float(_safe_get(row, "cur_rate"))
    if rate is None:
        return "A formal grid connection study would still be needed before any investment decision."
    if rate < 0.03:
        return "That curtailment signal is relatively low, so grid constraint pressure does not look like the main barrier at screening stage."
    if rate < 0.10:
        return "That curtailment signal is moderate: it does not rule the site out, but it should be included in project financial modelling."
    return "That curtailment signal is elevated and could materially affect project revenue, so grid due diligence should be treated as a priority."


def _development_implication(row: Dict[str, Any]) -> str:
    level = _risk_level(row)
    if level == "High":
        return (
            "Overall, this site is not automatically excluded, but it should be treated as a high-risk candidate. "
            "The next step is to verify grid capacity, test commercial feasibility, and prepare a bushfire risk management plan before proceeding."
        )
    if level == "Medium":
        return (
            "Overall, this site can remain under consideration, but it needs targeted due diligence before development decisions are made."
        )
    return (
        "Overall, this looks comparatively favourable at screening stage, while still requiring normal planning, fire-safety, and grid checks."
    )


# ---------------------------------------------------------------------------
# Data loading and sampling
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    print("[load] reading final_result.parquet ...")
    try:
        df = pd.read_parquet(FINAL_RESULT_PATH)
    except Exception:
        csv_alt = FINAL_RESULT_PATH.replace(".parquet", ".csv")
        if os.path.isfile(csv_alt):
            df = pd.read_csv(csv_alt)
        else:
            raise

    for candidate in ("pfi", "PFI", "property_id"):
        if candidate in df.columns:
            if candidate != "pfi":
                df = df.rename(columns={candidate: "pfi"})
            break

    keep = ["pfi", "risk_prob", "risk_score", "risk_level", *FEATURE_FIELD_NAMES]
    keep = [col for col in keep if col in df.columns]
    df = df[keep].rename(columns={"risk_prob": "risk_probability"})
    df["pfi"] = pd.to_numeric(df["pfi"], errors="coerce")
    df = df.dropna(subset=["pfi", "risk_probability", "risk_level"])
    df["pfi"] = df["pfi"].astype(int)
    df["risk_probability"] = pd.to_numeric(df["risk_probability"], errors="coerce").clip(0.0, 1.0)
    df["risk_level"] = df["risk_level"].astype(str).str.strip().str.capitalize()
    df = df[df["risk_level"].isin(["Low", "Medium", "High"])]
    df = df.drop_duplicates(subset=["pfi"], keep="first").reset_index(drop=True)
    return df


def _balanced_sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    base = n // 3
    remainder = n % 3
    parts = []
    for idx, level in enumerate(("Low", "Medium", "High")):
        take = base + (1 if idx < remainder else 0)
        pool = df[df["risk_level"] == level]
        if len(pool) == 0 or take <= 0:
            continue
        replace = take > len(pool)
        parts.append(pool.iloc[rng.choice(len(pool), size=take, replace=replace)])
    if not parts:
        return []
    out = pd.concat(parts, ignore_index=True)
    return out.to_dict(orient="records")


def _load_sft_counts() -> Dict[str, int]:
    if os.path.isfile(SFT_LOG_PATH):
        try:
            with open(SFT_LOG_PATH, "r", encoding="utf-8") as f:
                log = json.load(f)
            counts = log.get("counts_by_type")
            if isinstance(counts, dict) and counts:
                return {k: int(v) for k, v in counts.items()}
        except Exception:
            pass
    return dict(SFT_COUNTS_FALLBACK)


def _scaled_counts(ratio: float, min_per_type: int = 20) -> Dict[str, int]:
    counts = _load_sft_counts()
    return {k: max(min_per_type, int(round(v * ratio))) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Chosen generators
# ---------------------------------------------------------------------------

def chosen_normal(row: Dict[str, Any], rng: np.random.Generator) -> str:
    style = int(rng.integers(0, 3))
    if style == 0:
        return _paragraphs(_headline(row), _fire_context(row, max_items=2))
    if style == 1:
        return _paragraphs(_headline(row), _fire_context(row, max_items=3))
    return _paragraphs(_headline(row), _fire_context(row, max_items=3), _development_implication(row))


def chosen_short(row: Dict[str, Any], rng: np.random.Generator) -> str:
    pfi = _safe_get(row, "pfi")
    level = _risk_level(row)
    pct = _fmt_pct(_safe_get(row, "risk_probability"))
    score = _fmt_score(row)
    return _pick(rng, [
        f"PFI {pfi} is {level} risk, with an estimated probability of {pct} and a composite risk score of {score}.",
        f"Short answer: PFI {pfi} is currently assessed as {level} risk ({pct}; risk score {score}).",
        f"The headline result for PFI {pfi} is {level} risk, probability {pct}, risk score {score}.",
    ])


def chosen_why(row: Dict[str, Any], rng: np.random.Generator) -> str:
    return _paragraphs(
        _headline(row),
        _fire_context(row, max_items=4),
        _structural_note(row),
        "In simple terms, the retrieved result reflects the broader site context, not just one isolated number."
    )


def chosen_detailed(row: Dict[str, Any], rng: np.random.Generator) -> str:
    return _paragraphs(
        _headline(row),
        _fire_context(row, max_items=5),
        _structural_note(row),
        _grid_context(row),
        _curtailment_implication(row),
    )


def chosen_simple(row: Dict[str, Any], rng: np.random.Generator) -> str:
    pfi = _safe_get(row, "pfi")
    level = _risk_level(row)
    pct = _fmt_pct(_safe_get(row, "risk_probability"))
    score = _fmt_score(row)
    if level == "Low":
        meaning = "this looks like a lower-risk site in the retrieved screening result"
    elif level == "Medium":
        meaning = "this is a watch-list site, not a clear rejection but not a clean low-risk option either"
    else:
        meaning = "this should be treated as a high-risk candidate that needs careful review"
    return _paragraphs(
        f"In plain English, PFI {pfi} means: {meaning}. The model estimates the risk probability at about {pct}, with a composite risk score of {score}.",
        "The practical point is to use this as an early screening signal, then check the fire, planning, and grid details before making a development decision."
    )


def chosen_advice(row: Dict[str, Any], rng: np.random.Generator) -> str:
    pfi = _safe_get(row, "pfi")
    level = _risk_level(row)
    first = f"For PFI {pfi}, I would treat the retrieved {level} result as a first-pass screening signal, not a final development decision."
    if level == "High":
        second = "The first priority should be to understand the fire-risk constraint, especially whether the site sits in a mapped bushfire-prone area or has weak emergency-response access."
        third = "Before committing to development, compare it with lower-risk candidates, verify grid connection capacity, and prepare a bushfire management plan."
    elif level == "Medium":
        second = "The site can stay under consideration, but the risk drivers should be reviewed before committing capital."
        third = "The useful next step is targeted due diligence: check the fire overlay, nearby response coverage, vegetation context, and grid curtailment assumptions."
    else:
        second = "The site looks comparatively favourable, but routine planning, grid, and fire-safety checks are still needed."
        third = "It is still worth confirming connection capacity and local planning requirements before treating it as development-ready."
    return _paragraphs(first, second, third)


def chosen_curtailment(row: Dict[str, Any], rng: np.random.Generator) -> str:
    pfi = _safe_get(row, "pfi")
    capacity = _to_float(_safe_get(row, "P_project"))
    curtail = _to_float(_safe_get(row, "curtailment"))
    rate = _to_float(_safe_get(row, "cur_rate"))
    level = _risk_level(row)
    pct = _fmt_pct(_safe_get(row, "risk_probability"))
    score = _fmt_score(row)
    if capacity is None or curtail is None or rate is None:
        return f"The retrieved record for PFI {pfi} does not contain enough grid or curtailment data to assess network constraint risk."
    return _paragraphs(
        f"For PFI {pfi}, the grid result is based on zone substation load and import/export headroom data.",
        f"The recommended project capacity is about {capacity:.2f} MW. The estimated curtailed energy is {curtail:.2f} MWh, with a curtailment rate of {rate:.4f}%.",
        _curtailment_implication(row),
        f"The overall retrieved risk result is {level} ({pct}; risk score {score})."
    )


def chosen_siting(row: Dict[str, Any], rng: np.random.Generator) -> str:
    pfi = _safe_get(row, "pfi")
    level = _risk_level(row)
    pct = _fmt_pct(_safe_get(row, "risk_probability"))
    score = _fmt_score(row)
    is_prone = _to_float(_safe_get(row, "is_prone"))
    capacity = _to_float(_safe_get(row, "P_project"))
    curtail = _to_float(_safe_get(row, "curtailment"))
    rate = _to_float(_safe_get(row, "cur_rate"))

    if level == "High":
        opening = f"For PFI {pfi}, the retrieved assessment suggests the site is not automatically excluded, but it should be treated as a high-risk candidate that needs further technical review."
    elif level == "Medium":
        opening = f"For PFI {pfi}, the retrieved assessment suggests the site may remain under consideration, but it needs targeted due diligence before development."
    else:
        opening = f"For PFI {pfi}, the retrieved assessment suggests the site is a relatively favourable candidate at screening stage."

    fire = "From the bushfire side, "
    if is_prone is not None and is_prone >= 0.5:
        fire += "the site intersects a mapped bushfire-prone area, so permitting, mitigation, and emergency planning would need careful attention."
    else:
        fire += "the site is not mapped as bushfire-prone, which is a favourable sign, though standard fire-safety review is still needed."

    if capacity is not None and curtail is not None and rate is not None:
        grid = (
            f"The grid result is also important. The model recommends a project capacity of around {capacity:.2f} MW. "
            f"Based on nearby zone substation load and import/export headroom, estimated curtailed energy is {curtail:.2f} MWh "
            f"with a curtailment rate of {rate:.4f}%."
        )
    else:
        grid = "The retrieved record does not contain enough grid data, so a formal connection study would be needed."

    close = f"This corresponds to a {level} risk site ({pct}; risk score {score}). {_development_implication(row)}"
    return "\n\n".join([opening, fire, grid, close])


def chosen_dual(row: Dict[str, Any], rng: np.random.Generator) -> str:
    pfi = _safe_get(row, "pfi")
    return _paragraphs(
        f"PFI {pfi} should be read across two dimensions: bushfire exposure and grid or curtailment constraint.",
        _headline(row),
        _fire_context(row, max_items=4),
        _structural_note(row),
        _grid_context(row),
        _curtailment_implication(row),
        _development_implication(row),
    )


def chosen_methodology(row: Dict[str, Any], rng: np.random.Generator) -> str:
    pfi = _safe_get(row, "pfi")
    level = _risk_level(row)
    pct = _fmt_pct(_safe_get(row, "risk_probability"))
    score = _fmt_score(row)
    return (
        f"The result for PFI {pfi} is {level} risk, with an estimated probability of {pct} and a composite risk score of {score}. "
        "This result is produced by the project's machine learning component, not guessed by the language model.\n\n"
        "Specifically, the system uses a LightGBM gradient-boosted decision tree model, which is suitable for structured tabular data and can capture non-linear relationships between bushfire exposure, site characteristics, and grid or curtailment constraints.\n\n"
        "The model considers two main dimensions: bushfire risk and grid or curtailment risk. Bushfire-related inputs include mapped bushfire-prone areas, vegetation indicators, historical fire records, and distance to fire-response facilities. Grid-related inputs include zone substation load profiles, import and export limits, project capacity, storage interaction, and curtailment indicators.\n\n"
        "All risk results are pre-computed and stored in a database indexed by PFI. When you ask about a property, the system retrieves the stored record and the language model explains it in plain language. The LLM does not re-predict the risk from the visible features in the prompt.\n\n"
        "This should be treated as a first-pass screening result, not a substitute for engineering review, fire authority consultation, planning assessment, or a formal grid connection study."
    )


def chosen_closing(_row: Optional[Dict[str, Any]], rng: np.random.Generator) -> str:
    return _pick(rng, [
        "You're welcome. If you have another site to check, send the PFI and I can retrieve the record.",
        "Glad I could help. Come back any time if you want to compare another site or revisit the assumptions.",
        "No problem. Good luck with the project, and feel free to ask if you need another risk screen.",
        "Anytime. Risk screening is iterative, so you can return whenever the project scope or site list changes.",
    ])


def chosen_not_found(fake_pfi: int, rng: np.random.Generator) -> str:
    return _pick(rng, [
        f"I could not find a retrieved property record for PFI {fake_pfi}. Please check the identifier or provide another PFI before I give a risk assessment.",
        f"No matching record was found for PFI {fake_pfi}. I should not infer a risk level without a stored database record.",
        f"PFI {fake_pfi} is not available in the current lookup database, so I cannot provide a property-specific risk result.",
    ])


def chosen_no_pfi(_row: Optional[Dict[str, Any]], rng: np.random.Generator) -> str:
    return _pick(rng, [
        "I need a PFI or another location reference before I can retrieve a site-specific risk result.",
        "Please provide the PFI, postcode, suburb, or coordinates first. Without a lookup key, I would be guessing.",
        "I can help, but I need to know which property to retrieve. A PFI is the best identifier.",
    ])


# ---------------------------------------------------------------------------
# Rejected generators
# ---------------------------------------------------------------------------

def rejected_mechanical(row: Dict[str, Any], rng: np.random.Generator) -> str:
    pfi = _safe_get(row, "pfi")
    level = _risk_level(row)
    pct = _fmt_pct(_safe_get(row, "risk_probability"))
    score = _fmt_score(row)
    return _pick(rng, [
        f"Risk level: {level}. Probability: {pct}. Score: {score}. PFI: {pfi}.",
        f"PFI {pfi} — {level} risk ({pct}, score {score}).",
        f"Result: {level}. Probability {pct}, score {score}. See retrieved record for PFI {pfi}.",
    ])


def rejected_bullet_dump(row: Dict[str, Any], rng: np.random.Generator) -> str:
    pfi = _safe_get(row, "pfi")
    level = _risk_level(row)
    pct = _fmt_pct(_safe_get(row, "risk_probability"))
    score = _fmt_score(row)
    veg = _to_float(_safe_get(row, "veg_area"))
    prone = _safe_get(row, "is_prone")
    fire = _safe_get(row, "fire_count")
    yrs = _safe_get(row, "yrs_since_last_burn")
    cap = _safe_get(row, "P_project")
    cur = _safe_get(row, "curtailment")
    rate = _safe_get(row, "cur_rate")
    return (
        f"PFI {pfi}: {level} risk, probability {pct}, score {score}. "
        f"Feature context: bushfire-prone-area intersection = {prone}; native vegetation cover indicator = {veg}; "
        f"historical cumulative fire count = {fire}; years since last burn = {yrs}; project capacity = {cap}; "
        f"curtailed energy = {cur}; curtailment rate = {rate}. Overall expert interpretation: the retrieved values indicate the assigned risk level."
    )


def rejected_subtle_factual_error(row: Dict[str, Any], rng: np.random.Generator) -> str:
    pfi = _safe_get(row, "pfi")
    true_level = _risk_level(row)
    wrong_level = _opposite_level(true_level, rng)
    pct = _fmt_pct(_safe_get(row, "risk_probability"))
    score = _fmt_score(row)
    cap = _to_float(_safe_get(row, "P_project"))
    rate = _to_float(_safe_get(row, "cur_rate"))
    wrong_rate = None if rate is None else max(0.0, rate * _pick(rng, [0.1, 5.0, 10.0]))
    sentence = (
        f"PFI {pfi} is best read as {wrong_level} risk, with estimated probability {pct} and risk score {score}. "
        "The site does not raise major screening concerns and can generally progress with routine due diligence."
    )
    if cap is not None and wrong_rate is not None:
        sentence += f" A project capacity of {cap:.2f} MW is suitable, and the curtailment rate is approximately {wrong_rate:.4f}%."
    return sentence


def rejected_wrong_type_methodology(row: Dict[str, Any], rng: np.random.Generator) -> str:
    return _pick(rng, [
        "The system uses a LightGBM model trained on bushfire and grid features to produce risk probabilities. "
        "The result is pre-computed and stored in a database indexed by PFI.",
        "Risk is calculated using machine learning on structured tabular data. "
        "The model considers bushfire exposure, vegetation, and curtailment indicators.",
        "This result is generated by a gradient-boosted decision tree. "
        "The LLM retrieves pre-computed records and explains them in plain language.",
    ])


def rejected_methodology_as_risk(row: Dict[str, Any], rng: np.random.Generator) -> str:
    return rejected_bullet_dump(row, rng)


def rejected_closing_as_risk(row: Dict[str, Any], rng: np.random.Generator) -> str:
    return rejected_mechanical(row, rng) + " If you have more questions, feel free to ask."


def rejected_vague_no_data(rng: np.random.Generator) -> str:
    return _pick(rng, [
        "I cannot answer this fully from the information provided. Please provide more details.",
        "Sorry, I don't have enough information to help with that.",
        "Unable to process your request without additional context.",
        "I need more details before I can give you an accurate answer.",
    ])


def choose_rejected(row: Dict[str, Any], qtype: str, rng: np.random.Generator) -> str:
    if qtype == "methodology":
        return rejected_methodology_as_risk(row, rng)
    if qtype in {"siting_decision", "dual_risk", "curtailment_focus"} and int(rng.integers(0, 3)) == 0:
        return rejected_wrong_type_methodology(row, rng)
    if int(rng.integers(0, 100)) < 45:
        return rejected_subtle_factual_error(row, rng)
    if qtype in {"detailed_drivers", "why_explanation", "advice", "siting_decision", "dual_risk"}:
        return rejected_bullet_dump(row, rng)
    return rejected_mechanical(row, rng)


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

TYPE_CONFIG: Dict[str, Tuple[List[str], List[str], Callable, Callable]] = {
    "normal_query": (NORMAL_QUERY_TEMPLATES, INSTRUCTION_NORMAL_POOL, chosen_normal, choose_rejected),
    "short_answer": (SHORT_QUERY_TEMPLATES, INSTRUCTION_SHORT_POOL, chosen_short, choose_rejected),
    "detailed_drivers": (DETAIL_QUERY_TEMPLATES, INSTRUCTION_DETAIL_POOL, chosen_detailed, choose_rejected),
    "why_explanation": (WHY_QUERY_TEMPLATES, INSTRUCTION_WHY_POOL, chosen_why, choose_rejected),
    "advice": (ADVICE_QUERY_TEMPLATES, INSTRUCTION_ADVICE_POOL, chosen_advice, choose_rejected),
    "simple_explanation": (SIMPLE_QUERY_TEMPLATES, INSTRUCTION_SIMPLE_POOL, chosen_simple, choose_rejected),
    "curtailment_focus": (CURTAILMENT_QUERY_TEMPLATES, INSTRUCTION_CURTAILMENT_POOL, chosen_curtailment, choose_rejected),
    "siting_decision": (SITING_QUERY_TEMPLATES, INSTRUCTION_SITING_POOL, chosen_siting, choose_rejected),
    "dual_risk": (DUAL_RISK_QUERY_TEMPLATES, INSTRUCTION_DUAL_RISK_POOL, chosen_dual, choose_rejected),
    "methodology": (METHODOLOGY_QUERY_TEMPLATES, INSTRUCTION_METHODOLOGY_POOL, chosen_methodology, choose_rejected),
}


def make_data_pair(qtype: str, row: Dict[str, Any], rng: np.random.Generator) -> Dict[str, str]:
    queries, instructions, chosen_fn, rejected_fn = TYPE_CONFIG[qtype]
    question = _fmt_query(_pick(rng, queries), row)
    retrieved, _missing, _fmt_stats = render_retrieved_data(row)
    input_block = compose_input(question, retrieved)
    instruction = _pick(rng, instructions)
    chosen = chosen_fn(row, rng)
    rejected = rejected_fn(row, qtype, rng)
    return _record(instruction, input_block, chosen, rejected)


def make_closing_pair(row: Dict[str, Any], rng: np.random.Generator) -> Dict[str, str]:
    question = _pick(rng, CLOSING_QUERY_TEMPLATES)
    input_block = compose_no_data_input(question, reason="The user is ending the conversation; no property data is needed.")
    return _record(
        _pick(rng, INSTRUCTION_CLOSING_POOL),
        input_block,
        chosen_closing(None, rng),
        rejected_closing_as_risk(row, rng),
    )


def make_not_found_pair(fake_pfi: int, rng: np.random.Generator) -> Dict[str, str]:
    row = {"pfi": fake_pfi}
    question = _fmt_query(_pick(rng, NOT_FOUND_QUERY_TEMPLATES), row)
    input_block = compose_no_data_input(question, reason=f"No matching property record was found for PFI {fake_pfi}.")
    return _record(
        _pick(rng, INSTRUCTION_NOT_FOUND_POOL),
        input_block,
        chosen_not_found(fake_pfi, rng),
        rejected_vague_no_data(rng),
    )


def make_no_pfi_pair(rng: np.random.Generator) -> Dict[str, str]:
    question = _pick(rng, NO_PFI_QUERY_TEMPLATES)
    input_block = compose_no_data_input(question, reason="The user did not provide a PFI, postcode, suburb, or coordinates.")
    return _record(
        _pick(rng, INSTRUCTION_NO_PFI_POOL),
        input_block,
        chosen_no_pfi(None, rng),
        rejected_vague_no_data(rng),
    )


def build_pairs(df: pd.DataFrame, counts: Dict[str, int], rng: np.random.Generator) -> Tuple[List[Dict[str, str]], Counter]:
    records: List[Dict[str, str]] = []
    actual = Counter()
    real_pfis = set(df["pfi"].astype(int).tolist())

    for qtype in TYPE_CONFIG:
        for row in _balanced_sample(df, counts.get(qtype, 0), rng):
            records.append(make_data_pair(qtype, row, rng))
            actual[qtype] += 1

    closing_rows = _balanced_sample(df, counts.get("closing", 0), rng)
    for row in closing_rows:
        records.append(make_closing_pair(row, rng))
        actual["closing"] += 1

    used_fake: set[int] = set()
    target_nf = counts.get("not_found", 0)
    attempts = 0
    while actual["not_found"] < target_nf and attempts < target_nf * 50 + 100:
        attempts += 1
        fake = int(rng.integers(900_000_000, 999_999_999))
        if fake in real_pfis or fake in used_fake:
            continue
        used_fake.add(fake)
        records.append(make_not_found_pair(fake, rng))
        actual["not_found"] += 1

    for _ in range(counts.get("no_pfi", 0)):
        records.append(make_no_pfi_pair(rng))
        actual["no_pfi"] += 1

    order = rng.permutation(len(records)).tolist()
    shuffled = [records[i] for i in order]
    before = len(shuffled)
    shuffled = [r for r in shuffled if r["chosen"].strip() != r["rejected"].strip()]
    dropped = before - len(shuffled)
    if dropped:
        print(f"[filter] removed {dropped} pairs where chosen == rejected")
    return shuffled, actual


def validate(records: List[Dict[str, str]]) -> Dict[str, int]:
    return {
        "empty_fields": sum(1 for r in records for k in ("instruction", "input", "chosen", "rejected") if not str(r.get(k, "")).strip()),
        "chosen_equals_rejected": sum(1 for r in records if r["chosen"].strip() == r["rejected"].strip()),
        "chosen_with_paragraph_breaks": sum(1 for r in records if "\n\n" in r["chosen"]),
        "rejected_with_paragraph_breaks": sum(1 for r in records if "\n\n" in r["rejected"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build diverse DPO preference dataset")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--ratio", type=float, default=0.5, help="DPO:SFT ratio; default 0.5 means 1:2")
    parser.add_argument("--seed", type=int, default=42001)
    parser.add_argument("--min-per-type", type=int, default=20)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = load_data()
    source_counts = {lvl: int((df["risk_level"] == lvl).sum()) for lvl in ("Low", "Medium", "High")}
    print(f"[load] source rows: {source_counts}")

    counts = _scaled_counts(args.ratio, args.min_per_type)
    records, actual_counts = build_pairs(df, counts, rng)
    checks = validate(records)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log = {
        "output_path": args.out,
        "total_pairs": len(records),
        "target_ratio_to_sft": args.ratio,
        "target_counts_by_type": counts,
        "actual_counts_by_type": dict(actual_counts),
        "source_rows_by_risk_level": source_counts,
        "validation": checks,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    with open(args.log, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"[write] {len(records)} DPO pairs -> {args.out}")
    print(f"[write] log -> {args.log}")
    print(f"[check] {checks}")
    print("\nLlamaFactory dataset_info.json snippet:")
    print(json.dumps({
        "risk_chat_dpo": {
            "file_name": "dpo_data.jsonl",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        }
    }, indent=2))


if __name__ == "__main__":
    main()
