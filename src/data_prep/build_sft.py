#!/usr/bin/env python3
"""
build_sft.py

Generates LlamaFactory Alpaca-style (instruction/input/output) SFT data for the
dual-risk renewable energy chatbot: bushfire site screening + power curtailment risk.

Design principles:
- RAG-style: model explains retrieved structured data, never recites memorised PFI→risk mappings.
- risk_probability / risk_level must appear in the retrieved input block; output stays grounded.
- Eleven sample types covering lookup, explanation, advice, curtailment analysis, siting
  decisions, and dual-risk assessment to produce diverse, expert-quality training data.

Sample types:
  Stratified by risk level (normal/short/detail splits):
    normal_query      Standard risk lookup answer
    short_answer      One-sentence risk summary
    detailed_drivers  Deep-dive on risk factors

  Balanced across risk levels:
    why_explanation   Explain why this risk rating was assigned
    advice            Preparedness and next-steps guidance
    simple_explanation Plain-language restatement
    curtailment_focus  Grid and power curtailment risk analysis   [NEW]
    siting_decision    Go/no-go site-screening recommendation     [NEW]
    dual_risk          Combined fire + grid structured assessment [NEW]

  No-data stubs:
    not_found         PFI absent from the database
    no_pfi            User supplied no property identifier
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
from sft_input_text import FEATURE_FIELD_NAMES, compose_input, compose_no_data_input, render_retrieved_data
import sft_output_text as out
from sft_diverse_samples import HAND_CRAFTED_QA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FINAL_RESULT_PATH = str(DATA_DIR / "final_result.parquet")
DEFAULT_OUT = str(OUTPUTS_DIR / "sft_data.jsonl")
DEFAULT_LOG = str(OUTPUTS_DIR / "sft_data_log.json")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_get(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        v = row.get(key, default)
    else:
        try:
            v = row[key] if key in row.index else default
        except Exception:
            v = getattr(row, key, default)
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    return v


def _risk_level(row: Dict[str, Any]) -> str:
    level = _safe_get(row, "risk_level")
    if level is not None:
        text = str(level).strip().capitalize()
        if text in {"Low", "Medium", "High"}:
            return text
    return "Unknown"


def _pick(rng: np.random.Generator, pool: List[str]) -> str:
    return pool[int(rng.integers(0, len(pool)))]


def _fmt_query(template: str, **kwargs: Any) -> str:
    try:
        return template.format(**kwargs)
    except KeyError:
        result = template
        for k, v in kwargs.items():
            result = result.replace("{" + k + "}", str(v))
        return result


def _merge_stats(fmt_stats: Dict[str, int], gen_stats: Dict[str, bool]) -> Dict[str, int]:
    return {
        "locality_lines_removed": int(fmt_stats.get("locality_lines_removed", 0)),
        "unknown_lines_omitted": int(fmt_stats.get("unknown_lines_omitted", 0)),
        "calibrated_bridge_used": int(bool(gen_stats.get("calibrated_bridge_used", False))),
        "mostly_low_medium_high": int(bool(gen_stats.get("mostly_low_medium_high", False))),
    }


SampleResult = Tuple[Dict[str, str], str, List[str], Dict[str, int]]

# ---------------------------------------------------------------------------
# Sample-type configuration table
# ---------------------------------------------------------------------------
# Each entry: (query_templates, instruction_pool, answer_fn, query_vars)
# query_vars: variable names injected into the query template ("pfi", "risk_level")

_SAMPLE_CONFIG: Dict[str, Tuple[List[str], List[str], Callable, Tuple[str, ...]]] = {
    "normal_query":       (NORMAL_QUERY_TEMPLATES,       INSTRUCTION_NORMAL_POOL,       out.generate_normal_answer,       ("pfi",)),
    "short_answer":       (SHORT_QUERY_TEMPLATES,         INSTRUCTION_SHORT_POOL,        out.generate_short_answer,        ("pfi",)),
    "why_explanation":    (WHY_QUERY_TEMPLATES,           INSTRUCTION_WHY_POOL,          out.generate_reason_answer,       ("pfi", "risk_level")),
    "detailed_drivers":   (DETAIL_QUERY_TEMPLATES,        INSTRUCTION_DETAIL_POOL,       out.generate_detailed_answer,     ("pfi",)),
    "curtailment_focus":  (CURTAILMENT_QUERY_TEMPLATES,   INSTRUCTION_CURTAILMENT_POOL,  out.generate_curtailment_answer,  ("pfi",)),
    "siting_decision":    (SITING_QUERY_TEMPLATES,        INSTRUCTION_SITING_POOL,       out.generate_siting_answer,       ("pfi",)),
    "dual_risk":          (DUAL_RISK_QUERY_TEMPLATES,     INSTRUCTION_DUAL_RISK_POOL,    out.generate_dual_risk_answer,    ("pfi",)),
    "advice":             (ADVICE_QUERY_TEMPLATES,        INSTRUCTION_ADVICE_POOL,       out.generate_advice_answer,       ()),
    "simple_explanation": (SIMPLE_QUERY_TEMPLATES,        INSTRUCTION_SIMPLE_POOL,       out.generate_simple_answer,       ()),
    "methodology":        (METHODOLOGY_QUERY_TEMPLATES,   INSTRUCTION_METHODOLOGY_POOL,  out.generate_methodology_answer,  ("pfi", "risk_level")),
}


def make_data_sample(
    type_key: str,
    row: Dict[str, Any],
    rng: np.random.Generator,
    show_unknown_fields: bool = False,
) -> SampleResult:
    queries, instructions, answer_fn, query_vars = _SAMPLE_CONFIG[type_key]
    kwargs: Dict[str, Any] = {}
    if "pfi" in query_vars:
        kwargs["pfi"] = _safe_get(row, "pfi")
    if "risk_level" in query_vars:
        kwargs["risk_level"] = _risk_level(row)
    user_q = _fmt_query(_pick(rng, queries), **kwargs)
    retrieved, missing, fmt_stats = render_retrieved_data(row, show_unknown_fields=show_unknown_fields)
    inp = compose_input(user_q, retrieved)
    answer, gen_stats = answer_fn(row, rng)
    instruction = _pick(rng, instructions)
    return (
        {"instruction": instruction, "input": inp, "output": answer},
        type_key,
        missing,
        _merge_stats(fmt_stats, gen_stats),
    )


def make_not_found_sample(fake_pfi: int, rng: np.random.Generator) -> SampleResult:
    user_q = _fmt_query(_pick(rng, NOT_FOUND_QUERY_TEMPLATES), pfi=fake_pfi)
    inp = compose_no_data_input(user_q, reason=f"No matching property record was found for PFI {fake_pfi}.")
    answer, gen_stats = out.generate_not_found_answer(fake_pfi, rng)
    instruction = _pick(rng, INSTRUCTION_NOT_FOUND_POOL)
    return ({"instruction": instruction, "input": inp, "output": answer}, "not_found", ["all"], _merge_stats({}, gen_stats))


def make_no_pfi_sample(rng: np.random.Generator) -> SampleResult:
    user_q = _pick(rng, NO_PFI_QUERY_TEMPLATES)
    inp = compose_no_data_input(user_q, reason="The user did not provide a PFI, postcode, suburb, or coordinates.")
    answer, gen_stats = out.generate_no_pfi_answer(user_q, rng)
    instruction = _pick(rng, INSTRUCTION_NO_PFI_POOL)
    return ({"instruction": instruction, "input": inp, "output": answer}, "no_pfi", ["all"], _merge_stats({}, gen_stats))


def make_closing_sample(rng: np.random.Generator) -> SampleResult:
    user_q = _pick(rng, CLOSING_QUERY_TEMPLATES)
    inp = compose_no_data_input(user_q, reason="The user is ending the conversation — no property data is needed.")
    answer, gen_stats = out.generate_closing_answer(rng)
    instruction = _pick(rng, INSTRUCTION_CLOSING_POOL)
    return ({"instruction": instruction, "input": inp, "output": answer}, "closing", ["all"], _merge_stats({}, gen_stats))


# ---------------------------------------------------------------------------
# Data loading
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
            raise RuntimeError(f"Cannot read {FINAL_RESULT_PATH} (needs pyarrow/fastparquet or same-name CSV)")

    for candidate in ("pfi", "PFI", "property_id", "Property_ID"):
        if candidate in df.columns:
            if candidate != "pfi":
                df = df.rename(columns={candidate: "pfi"})
            break
    else:
        df = df.reset_index().rename(columns={df.index.name or "index": "pfi"})

    df["pfi"] = pd.to_numeric(df["pfi"], errors="coerce")
    df = df.dropna(subset=["pfi"])
    df["pfi"] = df["pfi"].astype(int)

    if "risk_prob" not in df.columns:
        raise RuntimeError("final_result.parquet: missing risk_prob column")
    if "risk_level" not in df.columns:
        raise RuntimeError("final_result.parquet: missing risk_level column")

    keep = ["pfi", "risk_prob", "risk_score", "risk_level"] + [c for c in FEATURE_FIELD_NAMES if c in df.columns]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].rename(columns={"risk_prob": "risk_probability"})
    df = df.drop_duplicates(subset=["pfi"], keep="first")

    df["risk_probability"] = pd.to_numeric(df["risk_probability"], errors="coerce")
    if df["risk_probability"].dropna().gt(1.0).any():
        df["risk_probability"] /= 100.0
    df["risk_probability"] = df["risk_probability"].clip(0.0, 1.0)
    df = df.dropna(subset=["risk_probability"])
    df["risk_level"] = df["risk_level"].astype(str).str.strip().str.capitalize()
    df = df[df["risk_level"].isin(["Low", "Medium", "High"])]
    return df


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def stratified_sample(
    df: pd.DataFrame,
    risk_level: str,
    n: int,
    rng: np.random.Generator,
    warnings: List[str],
) -> pd.DataFrame:
    pool = df[df["risk_level"] == risk_level]
    if len(pool) == 0:
        warnings.append(f"No rows for risk_level={risk_level}.")
        return pool.iloc[0:0]
    if len(pool) < n:
        warnings.append(f"Only {len(pool)} rows for risk_level={risk_level} (requested {n}); using all.")
        return pool.copy().reset_index(drop=True)
    return pool.iloc[rng.choice(len(pool), size=n, replace=False)].reset_index(drop=True)


def balanced_sample(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    warnings: List[str],
    label: str = "",
) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0]
    per = max(1, n // 3)
    parts = []
    for lvl in ("Low", "Medium", "High"):
        pool = df[df["risk_level"] == lvl]
        if len(pool) == 0:
            continue
        take = min(per, len(pool))
        parts.append(pool.iloc[rng.choice(len(pool), size=take, replace=False)])
    if not parts:
        warnings.append(f"[{label}] balanced sample produced 0 rows (requested {n}).")
        return df.iloc[0:0]
    cat = pd.concat(parts, ignore_index=True)
    if len(cat) > n:
        cat = cat.iloc[rng.choice(len(cat), size=n, replace=False)].reset_index(drop=True)
    elif len(cat) < n:
        warnings.append(f"[{label}] produced {len(cat)} rows (requested {n}).")
    return cat


# ---------------------------------------------------------------------------
# Core generation (shared by train and eval)
# ---------------------------------------------------------------------------

def _generate_samples(
    df: pd.DataFrame,
    counts: Dict[str, int],
    rng: np.random.Generator,
    show_unk: bool,
    real_pfis: set,
    include_handcrafted: bool = True,
) -> Tuple[List[Dict[str, str]], Counter, Counter, Counter, Dict[str, int], List[str]]:
    """Generate SFT samples and return (samples, counts_by_type, counts_by_level,
    missing_field_counts, agg_stats, warnings)."""

    warnings: List[str] = []

    # Stratified pools — each pool splits 1/3 normal / 1/3 short / 1/3 detail
    stratified = {
        lvl: stratified_sample(df, lvl, counts[f"n_{lvl.lower()}"], rng, warnings)
        for lvl in ("Low", "Medium", "High")
    }

    # Balanced pools for ancillary sample types
    balanced_pools: Dict[str, pd.DataFrame] = {
        "why_explanation":    balanced_sample(df, counts["n_followup"],    rng, warnings, "followup"),
        "advice":             balanced_sample(df, counts["n_advice"],       rng, warnings, "advice"),
        "simple_explanation": balanced_sample(df, counts["n_simple"],       rng, warnings, "simple"),
        "curtailment_focus":  balanced_sample(df, counts["n_curtailment"],  rng, warnings, "curtailment"),
        "siting_decision":    balanced_sample(df, counts["n_siting"],       rng, warnings, "siting"),
        "dual_risk":          balanced_sample(df, counts["n_dual_risk"],    rng, warnings, "dual_risk"),
        "methodology":        balanced_sample(df, counts["n_methodology"],  rng, warnings, "methodology"),
    }

    samples: List[Dict[str, str]] = []
    counts_by_type: Counter = Counter()
    counts_by_level: Counter = Counter()
    missing_field_counts: Counter = Counter()
    agg: Dict[str, int] = {k: 0 for k in ("locality_lines_removed", "unknown_lines_omitted",
                                            "calibrated_bridge_used", "mostly_low_medium_high")}

    def add(
        sample: Dict[str, str],
        stype: str,
        rlevel: Optional[str],
        missing: List[str],
        stats: Dict[str, int],
    ) -> None:
        samples.append(sample)
        counts_by_type[stype] += 1
        if rlevel:
            counts_by_level[rlevel] += 1
        for m in missing:
            missing_field_counts[m] += 1
        for k in agg:
            agg[k] += stats.get(k, 0)

    # Stratified: 1/3 normal_query / 1/3 short_answer / 1/3 detailed_drivers
    for lvl, pool in stratified.items():
        if len(pool) == 0:
            continue
        rows = [dict(r) for r in pool.to_dict(orient="records")]
        n = len(rows)
        rows = [rows[i] for i in rng.permutation(n)]
        n1, n2 = n // 3, n // 3
        for sub_type, sub_rows in [
            ("normal_query",    rows[:n1]),
            ("short_answer",    rows[n1:n1 + n2]),
            ("detailed_drivers", rows[n1 + n2:]),
        ]:
            for r in sub_rows:
                s, t, miss, stats = make_data_sample(sub_type, r, rng, show_unk)
                add(s, t, lvl, miss, stats)

    # Balanced pools (including methodology)
    for type_key, pool in balanced_pools.items():
        for r in pool.to_dict(orient="records"):
            rec = dict(r)
            s, t, miss, stats = make_data_sample(type_key, rec, rng, show_unk)
            add(s, t, _risk_level(rec), miss, stats)

    # not_found stubs — use PFI integers outside the real dataset range
    used_fakes: set = set()
    target_nf = int(counts["n_not_found"])
    attempts = 0
    while counts_by_type["not_found"] < target_nf and attempts < target_nf * 50 + 1000:
        attempts += 1
        fake = int(rng.integers(900_000_000, 999_999_999))
        if fake in real_pfis or fake in used_fakes:
            continue
        used_fakes.add(fake)
        s, t, miss, stats = make_not_found_sample(fake, rng)
        add(s, t, None, miss, stats)
    if counts_by_type["not_found"] < target_nf:
        warnings.append(
            f"Generated only {counts_by_type['not_found']} not_found samples (requested {target_nf})."
        )

    # no_pfi stubs
    for _ in range(int(counts["n_no_pfi"])):
        s, t, miss, stats = make_no_pfi_sample(rng)
        add(s, t, None, miss, stats)

    # closing stubs
    for _ in range(int(counts["n_closing"])):
        s, t, miss, stats = make_closing_sample(rng)
        add(s, t, None, miss, stats)

    # Hand-crafted diverse samples (training only)
    if include_handcrafted:
        for hc in HAND_CRAFTED_QA:
            row_data = hc.get("row")
            question = hc["question"]
            if row_data is not None:
                retrieved, missing_f, fmt_stats = render_retrieved_data(row_data)
                inp = compose_input(question, retrieved)
                rlevel = str(row_data.get("risk_level", "")).strip().capitalize()
                if rlevel not in {"Low", "Medium", "High"}:
                    rlevel = None
            else:
                inp = compose_no_data_input(
                    question,
                    reason=hc.get("reason", "The user did not provide a PFI or location reference."),
                )
                missing_f = ["all"]
                fmt_stats = {}
                rlevel = None
            s = {"instruction": hc["instruction"], "input": inp, "output": hc["answer"]}
            add(s, hc.get("type", "diverse"), rlevel, missing_f, _merge_stats(fmt_stats, {}))

    # Shuffle to avoid type-ordered sequences for LlamaFactory
    samples = [samples[i] for i in rng.permutation(len(samples))]

    return samples, counts_by_type, counts_by_level, missing_field_counts, agg, warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dual-risk renewable energy chatbot SFT dataset (Alpaca JSONL)."
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSONL path")
    parser.add_argument("--log", default=DEFAULT_LOG, help="Output log JSON path")
    parser.add_argument("--n-low",          type=int, default=2000, help="Low risk stratified pool size")
    parser.add_argument("--n-medium",       type=int, default=2000, help="Medium risk stratified pool size")
    parser.add_argument("--n-high",         type=int, default=2000, help="High risk stratified pool size")
    parser.add_argument("--n-not-found",    type=int, default=300,  help="not_found stub count")
    parser.add_argument("--n-no-pfi",       type=int, default=300,  help="no_pfi stub count")
    parser.add_argument("--n-followup",     type=int, default=1000, help="why_explanation sample count")
    parser.add_argument("--n-advice",       type=int, default=500,  help="advice sample count")
    parser.add_argument("--n-simple",       type=int, default=700,  help="simple_explanation sample count")
    parser.add_argument("--n-curtailment",  type=int, default=600,  help="curtailment_focus sample count")
    parser.add_argument("--n-siting",       type=int, default=500,  help="siting_decision sample count")
    parser.add_argument("--n-dual-risk",    type=int, default=500,  help="dual_risk sample count")
    parser.add_argument("--n-methodology",  type=int, default=400,  help="methodology sample count")
    parser.add_argument("--n-closing",      type=int, default=300,  help="closing stub count")
    parser.add_argument("--seed",           type=int, default=90037)
    parser.add_argument("--show-unknown-fields", action="store_true", default=False,
                        help="Render 'xxx: Unknown' lines for missing optional fields.")
    # Validation / eval set
    parser.add_argument("--n-eval",   type=int, default=0,
                        help="Total samples in the eval JSONL (0 = skip). Proportionally mirrors training counts.")
    parser.add_argument("--eval-out", default=str(OUTPUTS_DIR / "sft_eval.jsonl"),
                        help="Output path for the eval JSONL")
    parser.add_argument("--eval-seed", type=int, default=None,
                        help="RNG seed for eval set (default: --seed + 1)")
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    warnings: List[str] = []
    show_unk = bool(args.show_unknown_fields)

    df = load_data()
    source_counts = {lvl: int((df["risk_level"] == lvl).sum()) for lvl in ("Low", "Medium", "High")}
    print(f"[load] source rows: {source_counts}")

    real_pfis = {int(p) for p in df["pfi"]}

    # ------------------------------------------------------------------
    # Training counts
    # ------------------------------------------------------------------
    train_counts = {
        "n_low":         args.n_low,
        "n_medium":      args.n_medium,
        "n_high":        args.n_high,
        "n_not_found":   args.n_not_found,
        "n_no_pfi":      args.n_no_pfi,
        "n_followup":    args.n_followup,
        "n_advice":      args.n_advice,
        "n_simple":      args.n_simple,
        "n_curtailment": args.n_curtailment,
        "n_siting":      args.n_siting,
        "n_dual_risk":   args.n_dual_risk,
        "n_methodology": args.n_methodology,
        "n_closing":     args.n_closing,
    }

    samples, counts_by_type, counts_by_level, missing_field_counts, agg, gen_warnings = _generate_samples(
        df=df,
        counts=train_counts,
        rng=rng,
        show_unk=show_unk,
        real_pfis=real_pfis,
        include_handcrafted=True,
    )
    warnings.extend(gen_warnings)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[write] {len(samples)} samples → {args.out}")

    log = {
        "output_path": args.out,
        "total_samples": len(samples),
        "counts_by_type": dict(counts_by_type),
        "counts_by_risk_level": dict(counts_by_level),
        "source_rows_by_risk_level": source_counts,
        "missing_field_counts": dict(missing_field_counts),
        "show_unknown_fields": show_unk,
        **{f"{k}_count": v for k, v in agg.items()},
        "seed": int(args.seed),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "args": vars(args),
        "warnings": warnings,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    with open(args.log, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"[write] log → {args.log}")

    # ------------------------------------------------------------------
    # Eval / validation set (different seed, scaled-down counts)
    # ------------------------------------------------------------------
    if args.n_eval > 0:
        eval_seed = args.eval_seed if args.eval_seed is not None else int(args.seed) + 1
        eval_rng = np.random.default_rng(eval_seed)
        total_train = max(len(samples), 1)
        scale = args.n_eval / total_train

        eval_counts = {k: max(1, round(v * scale)) for k, v in train_counts.items()}

        eval_samples, _, _, _, _, eval_warnings = _generate_samples(
            df=df,
            counts=eval_counts,
            rng=eval_rng,
            show_unk=show_unk,
            real_pfis=real_pfis,
            include_handcrafted=False,
        )
        if eval_warnings:
            print(f"[eval] warnings: {eval_warnings}")

        os.makedirs(os.path.dirname(os.path.abspath(args.eval_out)), exist_ok=True)
        with open(args.eval_out, "w", encoding="utf-8") as f:
            for s in eval_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[write] {len(eval_samples)} eval samples (seed={eval_seed}) → {args.eval_out}")

    snippet = {
        "risk_chat_sft": {
            "file_name": os.path.basename(args.out),
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    }
    print("\n=== LlamaFactory dataset_info.json snippet ===")
    print(json.dumps(snippet, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
