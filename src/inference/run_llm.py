#!/usr/bin/env python3
"""
run_llm.py — Dual-risk renewable energy chatbot (server inference).

Classifies the user's question into one of 11 SFT training types, retrieves
the matching property record from SQLite, assembles a prompt in exactly the
Alpaca format used during SFT training, and returns the model's answer.

Usage:
    python run_llm.py           # normal mode
    python run_llm.py --think   # Qwen3 chain-of-thought mode
"""

from __future__ import annotations

import re
import sqlite3
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the exact rendering functions used to build the SFT dataset so the
# retrieved-data block and task-note format match training exactly.
from src.data_prep.sft_input_text import compose_input, compose_no_data_input, render_retrieved_data
from src.data_prep.sft_instruction_text import (
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
)

# ---------------------------------------------------------------------------
# Server paths — adjust before deploying
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip("\"'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_config(path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        config: Dict[str, Any] = {}
        section: Optional[str] = None
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line:
                continue
            if not line.startswith(" ") and line.endswith(":"):
                section = line[:-1].strip()
                config[section] = {}
                continue
            if section and ":" in line:
                key, value = line.strip().split(":", 1)
                config[section][key.strip()] = _parse_scalar(value)
        return config


def _resolve_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


CONFIG_PATH = Path(os.environ.get("CAPSTONE_CONFIG", DEFAULT_CONFIG_PATH))
CONFIG = _load_config(CONFIG_PATH)
PATHS = CONFIG.get("paths", {})
GENERATION = CONFIG.get("generation", {})

DB_PATH = _resolve_path(os.environ.get("CAPSTONE_DB_PATH", PATHS.get("database", "data/sft_sql.sqlite")))
BASE_MODEL = _resolve_path(os.environ.get("CAPSTONE_BASE_MODEL", PATHS.get("base_model", "Qwen3-4B")))
LORA_PATH = _resolve_path(
    os.environ.get("CAPSTONE_LORA_PATH", PATHS.get("lora_checkpoint", "checkpoints/train_2026-05-23-20-15-30"))
)

MAX_NEW_TOKENS = int(GENERATION.get("max_new_tokens", 512))
TEMPERATURE = float(GENERATION.get("temperature", 0.2))
TOP_P = float(GENERATION.get("top_p", 0.9))
REPETITION_PENALTY = float(GENERATION.get("repetition_penalty", 1.05))

SYSTEM_PROMPT = (
    "You are a dual-risk analyst specialising in renewable energy site screening "
    "for Victoria, Australia. You assess both bushfire exposure and power curtailment "
    "/ grid constraint risk. Always base your answers strictly on the retrieved "
    "structured data provided in the input. Never invent risk probabilities, risk "
    "levels, capacity figures, or any field value not explicitly shown."
)

# ---------------------------------------------------------------------------
# Question-type classifier
# ---------------------------------------------------------------------------
# Ordered by specificity: more specific types first to avoid false matches.

_CLOSING_PHRASES: List[str] = [
    "thank", "thanks", "cheers", "that's all", "that's everything",
    "that's helpful", "that's what i needed", "no more questions",
    "goodbye", "bye", "see you", "take care", "all done", "i'm done",
    "we're done", "that'll do", "perfect, thanks", "great, thanks",
]

_TYPE_KEYWORDS: List[Tuple[str, List[str]]] = [
    # Most specific types first to avoid false matches on broad keywords.
    ("methodology", [
        "how do you know", "where does the data come from", "data source",
        "where does this come from", "how accurate", "is this accurate",
        "how reliable", "is it correct", "how correct", "how was this calculated",
        "how is this calculated", "what model", "which model", "what data",
        "how was the risk", "how is the risk", "how is the score", "how was the score",
        "trained on", "machine learning", "ml model", "lightgbm",
        "methodology", "how does the model", "where do you get",
        "based on what", "basis for", "evidence behind",
    ]),
    ("curtailment_focus", [
        "curtailment", "curtailed", "grid constraint", "grid risk", "grid export",
        "dispatch", "p_project", "project capacity", "energy storage",
        "grid congestion", "network constraint", "export limit", "curtailment rate",
        "curtailment figure", "grid and curtailment",
    ]),
    ("siting_decision", [
        "good candidate", "worth developing", "should i develop", "should i build",
        "go/no-go", "viable site", "siting", "site selection", "worth progressing",
        "development risk", "should i consider", "development candidate",
        "proceed with", "progress this site", "renewable project",
        "solar project", "wind project", "recommended for development",
    ]),
    ("dual_risk", [
        "both risks", "combined risk", "dual risk", "fire and grid",
        "fire and curtailment", "two risk", "two dimensions",
        "bushfire and grid", "bushfire and curtailment", "both dimensions",
        "physical and grid", "two-part", "both fire",
    ]),
    # simple / short before why_explanation to avoid "explain" false-matching
    ("simple_explanation", [
        "simple", "simply", "plain", "less technical", "not a specialist",
        "easy to understand", "explain simply", "plain english",
        "layman", "non-technical", "everyday language", "in simple terms",
        "without jargon", "what does this mean", "simpler words",
    ]),
    ("short_answer", [
        "quick", "brief", "short answer", "just the level", "just the risk",
        "one sentence", "in a sentence", "headline", "compact",
        "only need the", "just give me", "keep it brief",
    ]),
    # detailed_drivers before why_explanation: "factors" matches both, detailed is more specific
    ("detailed_drivers", [
        "detail", "detailed", "breakdown", "full breakdown",
        "full picture", "go into detail", "walk me through",
        "all the factors", "main factors", "key factors", "risk factors",
        "comprehensive", "in depth", "deep dive", "full analysis",
    ]),
    ("why_explanation", [
        "why", "reason", "reasoning", "how did", "what makes",
        "why is it", "why medium", "why high", "why low", "why isn't",
        "what in the data", "logic behind", "how come", "explain why",
        "explain the reason", "explain the rating", "explain the risk level",
    ]),
    ("advice", [
        "what should i do", "what should i", "prepare", "preparation",
        "preparedness", "advice", "precaution", "precautions",
        "pay attention", "recommended", "next steps", "what actions",
        "what can i do", "reduce risk", "safety", "mitigation",
        "how should i plan", "what steps",
    ]),
]

_INSTRUCTION_POOLS: Dict[str, List[str]] = {
    "normal_query":       INSTRUCTION_NORMAL_POOL,
    "short_answer":       INSTRUCTION_SHORT_POOL,
    "detailed_drivers":   INSTRUCTION_DETAIL_POOL,
    "why_explanation":    INSTRUCTION_WHY_POOL,
    "curtailment_focus":  INSTRUCTION_CURTAILMENT_POOL,
    "siting_decision":    INSTRUCTION_SITING_POOL,
    "dual_risk":          INSTRUCTION_DUAL_RISK_POOL,
    "advice":             INSTRUCTION_ADVICE_POOL,
    "simple_explanation": INSTRUCTION_SIMPLE_POOL,
    "methodology":        INSTRUCTION_METHODOLOGY_POOL,
    "not_found":          INSTRUCTION_NOT_FOUND_POOL,
    "no_pfi":             INSTRUCTION_NO_PFI_POOL,
    "closing":            INSTRUCTION_CLOSING_POOL,
}


def classify_question(text: str) -> str:
    """Map user text to the closest SFT question type using word-boundary matching."""
    lower = text.lower()
    # Closing phrases are short social messages — check substring match (no word boundary needed)
    if any(phrase in lower for phrase in _CLOSING_PHRASES) and len(text.split()) <= 12:
        return "closing"
    for qtype, keywords in _TYPE_KEYWORDS:
        if any(re.search(r"\b" + re.escape(k) + r"\b", lower) for k in keywords):
            return qtype
    return "normal_query"


def _pick_instruction(qtype: str, seed_text: str) -> str:
    """Deterministically pick an instruction from the training pool (no RNG)."""
    pool = _INSTRUCTION_POOLS.get(qtype, INSTRUCTION_NORMAL_POOL)
    return pool[sum(ord(c) for c in seed_text) % len(pool)]


# ---------------------------------------------------------------------------
# PFI extraction
# ---------------------------------------------------------------------------

def extract_pfi(text: str) -> Optional[int]:
    # Explicit "PFI 12345" pattern
    m = re.search(r"\bpfi\s*[:#-]?\s*(\d{4,})\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Bare 6+ digit number treated as PFI
    m = re.search(r"\b(\d{6,})\b", text)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Database lookup
# ---------------------------------------------------------------------------

def get_risk_by_pfi(pfi: int) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM property_risk WHERE pfi = ? LIMIT 1", (pfi,)
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> Tuple[Any, Any]:
    print(f"[load] base model : {BASE_MODEL}")
    print(f"[load] LoRA path  : {LORA_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()
    print("[load] ready.\n")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(tokenizer: Any, model: Any, instruction: str, input_block: str, think: bool = False) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{instruction}\n\n{input_block}"},
    ]
    # Qwen3 supports enable_thinking; fall back gracefully for other models.
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=think
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=REPETITION_PENALTY,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()


# ---------------------------------------------------------------------------
# New-location detector — prevents context carry-forward on geographic queries
# ---------------------------------------------------------------------------

_NEW_LOCATION_SIGNALS: List[str] = [
    "area", "region", "suburb", "town", "city", "council", "district",
    "postcode", "zone", "near ", "around ", "in the ", "what about ",
    "how about ", "victoria", " vic ", "ballarat", "geelong", "bendigo",
    "shepparton", "warrnambool", "mildura", "wodonga", "horsham",
]


def _asks_new_location(text: str) -> bool:
    """Return True if the question looks like a new geographic query without a PFI."""
    lower = text.lower()
    if re.search(r"\b3\d{3}\b", lower):   # Victorian postcode pattern
        return True
    return any(sig in lower for sig in _NEW_LOCATION_SIGNALS)


# ---------------------------------------------------------------------------
# Chat — main routing logic
# ---------------------------------------------------------------------------

def chat(
    user_question: str,
    tokenizer: Any,
    model: Any,
    context_row: Optional[Dict[str, Any]] = None,
    think: bool = False,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Route the question, assemble the prompt, generate, and return
    (answer, updated_context_row).

    Context carry-forward: if no PFI is given but a previous lookup exists,
    the prior row is reused so follow-up questions (why / advice / simple)
    work without repeating the PFI.
    """
    # Closing check first — farewell messages need no PFI or context.
    if classify_question(user_question) == "closing":
        instruction = _pick_instruction("closing", user_question)
        input_block = compose_no_data_input(
            user_question,
            reason="The user is ending the conversation — no property data is needed.",
        )
        answer = generate(tokenizer, model, instruction, input_block, think=think)
        return answer, context_row

    pfi = extract_pfi(user_question)
    row = get_risk_by_pfi(pfi) if pfi is not None else None

    if pfi is not None and row is None:
        # PFI recognised but not in DB
        instruction = _pick_instruction("not_found", user_question)
        input_block  = compose_no_data_input(
            user_question,
            reason=f"No matching property record was found for PFI {pfi}.",
        )

    elif row is None and context_row is None:
        # No PFI and no prior context
        instruction = _pick_instruction("no_pfi", user_question)
        input_block  = compose_no_data_input(
            user_question,
            reason="The user did not provide a PFI, postcode, suburb, or coordinates.",
        )

    else:
        # If no PFI in this message but question is a new geographic query,
        # treat as no_pfi rather than silently reusing the previous property.
        if row is None and _asks_new_location(user_question):
            instruction = _pick_instruction("no_pfi", user_question)
            input_block = compose_no_data_input(
                user_question,
                reason="The user asked about a geographic area but did not provide a PFI.",
            )
            answer = generate(tokenizer, model, instruction, input_block, think=think)
            return answer, context_row  # preserve prior context in case user provides PFI next

        qtype = classify_question(user_question)

        # Fresh row or carry-forward from context
        effective = row if row is not None else context_row
        has_risk = effective.get("risk_probability") is not None and effective.get("risk_level") is not None

        if not has_risk:
            instruction = (
                "Answer using the retrieved property features, but note that the "
                "calibrated risk probability is unavailable. Do not invent risk values."
            )
        else:
            instruction = _pick_instruction(qtype, user_question)

        retrieved_text, _, _ = render_retrieved_data(effective)
        input_block = compose_input(user_question, retrieved_text)

    answer = generate(tokenizer, model, instruction, input_block, think=think)
    return answer, row if row is not None else context_row


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    think_mode = "--think" in sys.argv

    tokenizer, model = load_model()

    print("Dual-Risk Renewable Energy Chatbot")
    print("Type 'exit' to quit. Pass --think for chain-of-thought mode.")
    print("Examples:")
    print("  What is the bushfire risk for PFI 73461?")
    print("  Should I develop a solar project at PFI 73554?")
    print("  What are the curtailment figures for PFI 73477?")
    print("  Why is PFI 73544 High risk?")
    print("  Give me a combined fire and grid assessment for PFI 73557.")

    context_row: Optional[Dict[str, Any]] = None
    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        answer, context_row = chat(user_input, tokenizer, model, context_row, think=think_mode)
        print(f"\nAssistant:\n{answer}")
