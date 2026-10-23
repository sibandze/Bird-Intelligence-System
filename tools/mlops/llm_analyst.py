# tools/mlops/llm_analyst.py

"""
Optional LLM enrichment layer for the deterministic training report.

Contract:
    - enrich_report(report, config) -> dict, never raises.
    - report["llm"] is populated on return (with error set if something failed).
    - Caller is responsible for persisting the returned report to disk
      (report.json or elsewhere).
    - Deterministic fields in `report` are never mutated.

Trust boundary:
    - The deterministic diagnosis is authoritative. The LLM may express
      disagreement in `conflicts_with_deterministic`, but that field is
      visible-only — nothing downstream auto-applies it.
    - Experiments produced here are tagged `source: "llm"` and are
      candidates for human approval, never auto-applied.
"""

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.0-flash"
PROMPT_VERSION = "v1"
CALL_TIMEOUT_SEC = 30
TEMPERATURE = 0.2

PROMPT_V1 = """You are given a deterministic training diagnosis.
Rules:
- You MUST NOT regenerate numbers. Use only evidence from report.json.
- The deterministic diagnosis is authoritative. Only disagree if you see
  a pattern the rules cannot see (e.g. NaN pattern, per-class collapse).
- If you disagree, put it in conflicts_with_deterministic with explicit
  evidence references to fields in the report.
- Output JSON only, matching the requested schema.

Report: {report}
Config (sanitized): {config}
"""

# JSON schema for Gemini's response. Shapes are locked to prevent
# downstream KeyErrors. `from` / `to` are strings on purpose — see module
# docstring; hparam_advisor coerces against the real config schema.
_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "narrative": {"type": "string"},
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "supporting_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["text", "confidence", "supporting_evidence"],
            },
        },
        "experiments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "param": {"type": "string"},
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "rationale": {"type": "string"},
                    "expected_effect": {"type": "string"},
                },
                "required": ["param", "from", "to", "rationale", "expected_effect"],
            },
        },
        "conflicts_with_deterministic": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "deterministic_diagnosis": {"type": "string"},
                    "llm_diagnosis": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "deterministic_diagnosis",
                    "llm_diagnosis",
                    "evidence_refs",
                ],
            },
        },
    },
    "required": [
        "narrative",
        "hypotheses",
        "experiments",
        "conflicts_with_deterministic",
    ],
}


def _hash(d: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _deterministic_view(report: Dict[str, Any]) -> Dict[str, Any]:
    """Return the subset of the report used for caching. Excludes fields
    that vary per analysis run (`analyzed_at`) and any prior LLM block."""
    return {k: v for k, v in report.items() if k not in ("analyzed_at", "llm")}


def _cache_key(report: Dict[str, Any]) -> str:
    return _hash(
        {
            "model": MODEL_NAME,
            "prompt_hash": _hash({"prompt": PROMPT_V1, "version": PROMPT_VERSION}),
            "schema_hash": _hash(_RESPONSE_SCHEMA),
            "report": _deterministic_view(report),
        }
    )


def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    # Only forward fields the LLM should reason about. Do not pass paths,
    # API keys, wandb configs, etc.
    return {
        "training": config.get("training", {}),
        "model": {"type": config.get("model", {}).get("type")},
        "objective": config.get("experiment", {}).get("objective"),
    }


def _call_gemini(prompt: str) -> str:
    """Execute the Gemini call in a thread with a hard timeout."""
    from google import genai

    client = genai.Client()

    def _do():
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": _RESPONSE_SCHEMA,
                "temperature": TEMPERATURE,
            },
        )
        return resp.text

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_do)
        try:
            return future.result(timeout=CALL_TIMEOUT_SEC)
        except FutTimeout:
            raise RuntimeError(f"Gemini call exceeded {CALL_TIMEOUT_SEC}s")


def _empty_llm_block(error: Optional[str]) -> Dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "narrative": None,
        "hypotheses": [],
        "experiments": [],
        "conflicts_with_deterministic": [],
        "error": error,
    }


def enrich_report(
    report: Dict[str, Any],
    config: Dict[str, Any],
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Add an `llm` block to `report` with Gemini Flash analysis.

    Never raises. On any failure, `report["llm"]["error"]` is set and the
    other LLM fields are empty. Caller is responsible for persisting the
    returned report to disk if `report["llm"]` should survive restarts.
    """
    try:
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY not set")

        cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "bis"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{_cache_key(report)}.json"

        if cache_file.exists():
            report["llm"] = json.loads(cache_file.read_text())
            return report

        prompt = PROMPT_V1.format(
            report=json.dumps(_deterministic_view(report), default=str),
            config=json.dumps(_sanitize_config(config), default=str),
        )
        raw = _call_gemini(prompt)
        llm_data = json.loads(raw)

        # Provenance: tag experiments so the advisor can distinguish
        # rule-based from LLM-proposed. Never auto-applied downstream.
        for exp in llm_data.get("experiments", []):
            if isinstance(exp, dict):
                exp["source"] = "llm"

        llm_block = {
            "model": MODEL_NAME,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": _hash({"prompt": PROMPT_V1, "version": PROMPT_VERSION}),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **llm_data,
            "error": None,
        }

        try:
            cache_file.write_text(json.dumps(llm_block, indent=2))
        except OSError as e:
            logger.warning(f"LLM cache write failed ({cache_file}): {e}")

        report["llm"] = llm_block
        return report

    except Exception as e:
        logger.warning(f"LLM enrichment failed: {e}", exc_info=True)
        report["llm"] = _empty_llm_block(error=str(e))
        return report
