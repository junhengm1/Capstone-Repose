#!/usr/bin/env python3
"""
app.py — Flask web server for the Dual-Risk Renewable Energy Chatbot.

Wraps run_llm.chat() and serves the single-page frontend at GET /.

Endpoints:
  GET  /            → index.html
  POST /api/chat    → { message, session_id } → { response }
  POST /api/clear   → { session_id }          → { ok: true }

Run:
    python app.py
    python app.py --port 8080
    python app.py --think        # enable Qwen3 chain-of-thought
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.run_llm import chat, load_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

THINK_MODE  = "--think" in sys.argv
PORT        = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv)
                    if a == "--port" and i + 1 < len(sys.argv)), 5000)
HERE        = Path(__file__).resolve().parent
STATIC_DIR  = HERE / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR))

# Per-session context rows: { session_id: context_row | None }
_sessions: Dict[str, Optional[Dict[str, Any]]] = {}

# Load model once at startup (before first request)
print("[app] loading model…")
_tokenizer, _model = load_model()
print("[app] model ready — listening on port", PORT)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.post("/api/chat")
def api_chat():
    body = request.get_json(force=True, silent=True) or {}
    message    = (body.get("message") or "").strip()
    session_id = (body.get("session_id") or "").strip()

    if not message:
        return jsonify({"error": "Empty message."}), 400
    if not session_id:
        return jsonify({"error": "Missing session_id."}), 400

    context = _sessions.get(session_id)
    try:
        answer, new_context = chat(
            message, _tokenizer, _model,
            context_row=context, think=THINK_MODE
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    _sessions[session_id] = new_context
    return jsonify({"response": answer})


@app.post("/api/clear")
def api_clear():
    body = request.get_json(force=True, silent=True) or {}
    session_id = (body.get("session_id") or "").strip()
    _sessions.pop(session_id, None)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
