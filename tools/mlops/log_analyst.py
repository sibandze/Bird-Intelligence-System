# tools/mlops/log_analyst.py

"""
Deterministic training-log analyst.

Reads canonical training artifacts from a run directory and produces
structured diagnosis (report.json) + human-readable summary (report.md).

Design principles
-----------------
- Deterministic: no LLM, no randomness. Same inputs → same report.
- Objective-aware: supervised and SimCLR runs use different monitors,
  metrics, and diagnostic rules. Supervised overfit gap uses
  window-vs-window accuracy; SimCLR never diagnoses on contrastive acc.
- Evidence-first: every diagnosis carries the numbers that justify it.
  High accuracy alone is NEVER sufficient for "task too easy".
- Fail-safe: reporting failure must not fail the training run
  (AutoReportCallback swallows exceptions; _emit catches write errors).

Inputs (in run_dir)
-------------------
- training_metrics.json      (required; per-epoch history, JSON source of truth)
- evaluation_metrics.json    (optional; test-set metrics, from MetricsCollector)
- label_mappings.json        (optional; class names, for future use)

Outputs (in run_dir)
--------------------
- report.json   (structured, machine-readable, stable keys)
- report.md     (human-readable summary)

report.json is designed for diffing across runs. `analyzed_at` is the
only non-deterministic field; exclude it when hashing or comparing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Thresholds (single source of truth for diagnosis) ─────────────────
RECENT_WINDOW = 5
MIN_EPOCHS_FOR_STATS = 5
OVERFIT_GAP_THRESHOLD = 0.15
OVERFIT_MIN_TRAIN_ACC = 0.90
UNSTABLE_STD_THRESHOLD = 0.05
PLATEAU_EPOCHS = 8
EASY_VAL_ACC_THRESHOLD = 0.95
EASY_TRAIN_ACC_THRESHOLD = 0.99
EASY_GAP_THRESHOLD = 0.05
GRAD_NORM_WARN = 10.0


def _safe_float(x, default=None):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    if not rows:
        return "_no rows_"
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


class TrainingLogAnalyst:
    """
    Analyze one run directory and write report.json + report.md.

    Usage
    -----
    TrainingLogAnalyst(run_dir).analyze()          # programmatic
    python -m tools.mlops.log_analyst <run_dir>    # CLI
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.json_path = self.run_dir / "training_metrics.json"
        self.eval_path = self.run_dir / "evaluation_metrics.json"
        self.report_json_path = self.run_dir / "report.json"
        self.report_md_path = self.run_dir / "report.md"

    # ==================================================================
    # Public API
    # ==================================================================
    def analyze(self) -> Dict[str, Any]:
        try:
            df = self._load_history()
        except FileNotFoundError:
            return self._emit(self._error_report("training_metrics.json not found"))
        except Exception as e:
            return self._emit(self._error_report(f"failed to read history: {e}"))

        if df.empty:
            return self._emit(
                self._error_report(
                    "training history is empty", status="insufficient_data"
                )
            )

        objective = self._resolve_objective(df)
        test = self._load_test_metrics()
        common = self._common_analysis(df)

        if objective == "supervised_transformer":
            analysis = self._analyze_supervised(df, test, common)
        elif objective == "simclr":
            analysis = self._analyze_ssl(df, test, common)
        else:
            analysis = self._analyze_generic(df, test, common)

        report = {
            "run_id": self.run_dir.name,
            "objective": objective,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            **analysis,
        }
        return self._emit(report)

    # ==================================================================
    # I/O
    # ==================================================================
    def _emit(self, report: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.report_json_path.write_text(json.dumps(report, indent=2))
            self.report_md_path.write_text(self._render_markdown(report))
            diag = report.get("diagnosis", report.get("error", "?"))
            print(f"✓ Report written to {self.report_md_path} — {diag}")
        except Exception as e:
            logger.warning(f"Failed to write reports under {self.run_dir}: {e}")
        return report

    def _load_history(self) -> pd.DataFrame:
        if not self.json_path.exists():
            raise FileNotFoundError(str(self.json_path))
        data = json.loads(self.json_path.read_text())
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "epoch" not in df.columns:
            raise ValueError("training_metrics.json missing 'epoch' column")
        return df.sort_values("epoch").reset_index(drop=True)

    def _load_test_metrics(self) -> Dict[str, Any]:
        if not self.eval_path.exists():
            return {"present": False}
        try:
            data = json.loads(self.eval_path.read_text())
        except Exception as e:
            return {"present": False, "error": str(e)}
        if not isinstance(data, dict):
            return {"present": True, "raw": data}
        return {"present": True, **data}

    # ==================================================================
    # Objective resolution
    # ==================================================================
    def _resolve_objective(self, df: pd.DataFrame) -> str:
        # Preferred: explicit 'objective' column in the history
        if "objective" in df.columns:
            vals = df["objective"].dropna()
            if len(vals) > 0:
                return str(vals.iloc[-1])

        # Fallback: infer from column names
        cols = set(df.columns)
        if "val_contrastive_acc" in cols or "train_contrastive_acc" in cols:
            logger.warning(
                "Inferred objective=simclr from column names; "
                "add 'objective' to _build_logs for explicit typing."
            )
            return "simclr"
        if "val_acc" in cols and "train_acc" in cols:
            logger.warning(
                "Inferred objective=supervised_transformer from column names; "
                "add 'objective' to _build_logs for explicit typing."
            )
            return "supervised_transformer"

        logger.warning("Could not resolve objective; using generic analysis.")
        return "unknown"

    # ==================================================================
    # Shared analysis
    # ==================================================================
    def _common_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        epochs = df["epoch"].astype(int)
        attempts = (
            int(_safe_float(df.iloc[-1].get("attempt"), 0) or 0)
            if "attempt" in df.columns
            else 0
        )
        nan_cols = [c for c in df.columns if df[c].isna().any()]
        return {
            "history": {
                "n_epochs": int(n),
                "first_epoch": int(epochs.iloc[0]),
                "last_epoch": int(epochs.iloc[-1]),
                "attempts": attempts,
                "nan_metrics": nan_cols,
            },
            "short_history": n < MIN_EPOCHS_FOR_STATS,
        }

    def _best_epoch(self, df: pd.DataFrame, monitor: str, mode: str) -> Dict[str, Any]:
        """Locate the best epoch.

        Primary source: the last row's `best_epoch` field, which the
        trainer writes from its own _update_best logic and therefore
        agrees exactly with CheckpointCallback's criterion. Fallback:
        recompute from the monitor column.
        """
        if "best_epoch" in df.columns:
            be = _safe_float(df.iloc[-1].get("best_epoch"))
            if be is not None and be > 0:
                match = df[df["epoch"] == int(be)]
                if len(match) > 0:
                    row = match.iloc[0]
                    return {
                        "criterion": monitor,
                        "mode": mode,
                        "epoch": int(be),
                        "value": _safe_float(row.get(monitor)),
                        "train_loss": _safe_float(row.get("train_loss")),
                        "val_loss": _safe_float(row.get("val_loss")),
                        "val_acc": _safe_float(row.get("val_acc")),
                        "val_window_acc": _safe_float(row.get("val_window_acc")),
                        "val_contrastive_acc": _safe_float(
                            row.get("val_contrastive_acc")
                        ),
                        "source": "trainer_state",
                    }

        if monitor not in df.columns:
            return {
                "criterion": monitor,
                "mode": mode,
                "epoch": None,
                "value": None,
                "source": "unavailable",
            }

        series = df[["epoch", monitor]].dropna()
        if series.empty:
            return {
                "criterion": monitor,
                "mode": mode,
                "epoch": None,
                "value": None,
                "source": "all_nan",
            }

        idx = series[monitor].idxmax() if mode == "max" else series[monitor].idxmin()
        row = df.loc[idx]
        return {
            "criterion": monitor,
            "mode": mode,
            "epoch": int(row["epoch"]),
            "value": _safe_float(row[monitor]),
            "train_loss": _safe_float(row.get("train_loss")),
            "val_loss": _safe_float(row.get("val_loss")),
            "val_acc": _safe_float(row.get("val_acc")),
            "val_window_acc": _safe_float(row.get("val_window_acc")),
            "val_contrastive_acc": _safe_float(row.get("val_contrastive_acc")),
            "source": "recomputed",
        }

    def _recent(self, df: pd.DataFrame, col: str, n: int = RECENT_WINDOW) -> np.ndarray:
        if col not in df.columns:
            return np.array([], dtype=float)
        return df[col].dropna().tail(n).to_numpy(dtype=float)

    def _convergence(
        self, df: pd.DataFrame, monitor: str, mode: str, best_epoch: Optional[int]
    ) -> Dict[str, Any]:
        if monitor not in df.columns or best_epoch is None:
            return {
                "plateaued": False,
                "epochs_since_best": None,
                "improved_recently": None,
            }
        last_epoch = int(df["epoch"].iloc[-1])
        since = last_epoch - best_epoch
        # Improved recently if the best is within the recent window
        improved = since < RECENT_WINDOW
        return {
            "plateaued": since >= PLATEAU_EPOCHS,
            "epochs_since_best": int(since),
            "improved_recently": bool(improved),
        }

    def _stability(self, df: pd.DataFrame) -> Dict[str, Any]:
        recent_val_loss = self._recent(df, "val_loss")
        val_loss_std = (
            float(np.std(recent_val_loss)) if len(recent_val_loss) > 1 else None
        )
        val_loss_range = (
            float(np.ptp(recent_val_loss)) if len(recent_val_loss) > 1 else None
        )

        gn = self._recent(df, "avg_grad_norm", n=len(df))
        gn_mean = float(np.mean(gn)) if len(gn) > 0 else None
        gn_max = float(np.max(gn)) if len(gn) > 0 else None
        gn_p95 = float(np.percentile(gn, 95)) if len(gn) > 0 else None

        unstable = val_loss_std is not None and val_loss_std > UNSTABLE_STD_THRESHOLD
        grad_warn = gn_max is not None and gn_max > GRAD_NORM_WARN

        return {
            "val_loss_std_recent": val_loss_std,
            "val_loss_range_recent": val_loss_range,
            "grad_norm_mean": gn_mean,
            "grad_norm_max": gn_max,
            "grad_norm_p95": gn_p95,
            "unstable": bool(unstable),
            "grad_norm_warning": bool(grad_warn),
        }

    # ==================================================================
    # Supervised analysis
    # ==================================================================
    def _analyze_supervised(
        self, df: pd.DataFrame, test: Dict[str, Any], common: Dict[str, Any]
    ) -> Dict[str, Any]:
        monitor, mode = "val_acc", "max"
        best = self._best_epoch(df, monitor, mode)
        convergence = self._convergence(df, monitor, mode, best.get("epoch"))
        stability = self._stability(df)
        overfit = self._overfit_supervised(df)

        diagnosis, evidence, suggestions, warnings = self._diagnose_supervised(
            df, best, convergence, stability, overfit, common
        )

        return {
            "status": "ok",
            **common,
            "best": best,
            "convergence": convergence,
            "stability": stability,
            "overfit": overfit,
            "test": test,
            "diagnosis": diagnosis,
            "evidence": evidence,
            "suggestions": suggestions,
            "warnings": warnings,
            # Backward-compat scalar for downstream tools like
            # hparam_advisor.py (still mocked; will need updating to
            # read report["best"]["value"] + report["best"]["criterion"]).
            "compat": {"best_val_acc": best.get("value")},
        }

    def _overfit_supervised(self, df: pd.DataFrame) -> Dict[str, Any]:
        last = df.iloc[-1]
        train_acc = _safe_float(last.get("train_acc"))
        val_win = _safe_float(last.get("val_window_acc"))
        val_rec = _safe_float(last.get("val_acc"))

        # Apples-to-apples gap: window-level vs window-level
        gap_window = (
            train_acc - val_win
            if train_acc is not None and val_win is not None
            else None
        )
        # Recording-level benefit from averaging
        gap_agg = (
            val_rec - val_win if val_rec is not None and val_win is not None else None
        )

        # Primary signal falls back to recording-level gap only when
        # window-level is unavailable (older runs before val_window_acc).
        gap = (
            gap_window
            if gap_window is not None
            else (
                train_acc - val_rec
                if train_acc is not None and val_rec is not None
                else None
            )
        )

        suspect = (
            gap is not None
            and gap > OVERFIT_GAP_THRESHOLD
            and train_acc is not None
            and train_acc > OVERFIT_MIN_TRAIN_ACC
        )
        return {
            "train_acc": train_acc,
            "val_window_acc": val_win,
            "val_recording_acc": val_rec,
            "gap_window_vs_window": gap_window,
            "gap_recording_vs_window": gap_agg,
            "suspect": bool(suspect),
            "note": (
                "Overfit gap uses train_acc vs val_window_acc (window vs window). "
                "val_acc is recording-level and benefits from window averaging; "
                "it is reported separately as the recording-level accuracy."
            ),
        }

    def _diagnose_supervised(self, df, best, convergence, stability, overfit, common):
        evidence: List[str] = []
        suggestions: List[str] = []
        warnings: List[str] = []

        if common["short_history"]:
            warnings.append(
                f"History has < {MIN_EPOCHS_FOR_STATS} epochs; "
                f"windowed stability and convergence diagnostics are less reliable."
            )

        if common["history"]["nan_metrics"]:
            warnings.append(
                f"NaN values present in: {', '.join(common['history']['nan_metrics'])}"
            )

        if stability["unstable"]:
            evidence.append(
                f"val_loss std over last {RECENT_WINDOW} epochs = "
                f"{stability['val_loss_std_recent']:.4f} > {UNSTABLE_STD_THRESHOLD}"
            )
            suggestions += [
                "Reduce learning_rate by 50%",
                "Increase warmup_steps",
                "Enable or tighten gradient_clip (e.g. 1.0)",
            ]
            return "UNSTABLE_VAL", evidence, suggestions, warnings

        if overfit["suspect"]:
            evidence.append(
                f"train_acc={overfit['train_acc']:.3f} vs "
                f"val_window_acc={overfit['val_window_acc']:.3f} "
                f"(gap={overfit['gap_window_vs_window']:.3f} > {OVERFIT_GAP_THRESHOLD})"
            )
            suggestions += [
                "Increase spec_aug / acoustic_aug strength",
                "Increase dropout or weight_decay",
                "Evaluate on val_loss for early stopping instead of val_acc",
            ]
            return "OVERFIT", evidence, suggestions, warnings

        if self._suspected_too_easy(df):
            evidence.append(
                f"Sustained val_acc ≥ {EASY_VAL_ACC_THRESHOLD} and "
                f"train_acc ≥ {EASY_TRAIN_ACC_THRESHOLD} with small gap"
            )
            suggestions += [
                "Inspect for duplicate or near-duplicate recordings across splits",
                "Verify recording-level split (no window leakage between train/val)",
                "Consider harder augmentation or a stricter evaluation protocol",
            ]
            warnings.append(
                "High accuracy alone does NOT prove the task is too easy. "
                "Validate data integrity before changing the experimental setup."
            )
            return "TASK_TOO_EASY_SUSPECTED", evidence, suggestions, warnings

        if convergence["plateaued"]:
            evidence.append(
                f"No {best['criterion']} improvement for "
                f"{convergence['epochs_since_best']} epochs "
                f"(best={best['value']:.4f} @ ep {best['epoch']})"
            )
            suggestions.append(
                "Consider early stopping; verify the LR schedule reaches its floor"
            )
            return "PLATEAU", evidence, suggestions, warnings

        if best.get("value") is not None:
            evidence.append(
                f"Best {best['criterion']}={best['value']:.4f} @ epoch {best['epoch']}; "
                f"no instability, overfit, or plateau signals detected"
            )
        else:
            evidence.append("No monitor metric available for diagnosis")
        suggestions.append("Proceed to test evaluation / candidate for registry")
        return "HEALTHY", evidence, suggestions, warnings

    def _suspected_too_easy(self, df: pd.DataFrame) -> bool:
        last = df.iloc[-1]
        val_acc = _safe_float(last.get("val_acc"))
        train_acc = _safe_float(last.get("train_acc"))
        if val_acc is None or train_acc is None:
            return False
        if val_acc < EASY_VAL_ACC_THRESHOLD or train_acc < EASY_TRAIN_ACC_THRESHOLD:
            return False
        if abs(train_acc - val_acc) > EASY_GAP_THRESHOLD:
            return False
        # Require sustained high performance over the recent window
        recent = self._recent(df, "val_acc")
        if len(recent) < 3 or float(np.mean(recent)) < EASY_VAL_ACC_THRESHOLD:
            return False
        return True

    # ==================================================================
    # SimCLR analysis
    # ==================================================================
    def _analyze_ssl(
        self, df: pd.DataFrame, test: Dict[str, Any], common: Dict[str, Any]
    ) -> Dict[str, Any]:
        monitor, mode = "val_loss", "min"
        best = self._best_epoch(df, monitor, mode)
        convergence = self._convergence(df, monitor, mode, best.get("epoch"))
        stability = self._stability(df)

        diagnosis, evidence, suggestions, warnings = self._diagnose_ssl(
            df, best, convergence, stability, common
        )

        # Always attach: contrastive accuracy is not the arbiter
        warnings.append(
            "Contrastive accuracy is NOT downstream species accuracy. "
            "Compare SimCLR checkpoints via frozen-encoder linear probe."
        )

        return {
            "status": "ok",
            **common,
            "best": best,
            "convergence": convergence,
            "stability": stability,
            "test": test,
            "diagnosis": diagnosis,
            "evidence": evidence,
            "suggestions": suggestions,
            "warnings": warnings,
            "compat": {"best_val_loss": best.get("value")},
        }

    def _diagnose_ssl(self, df, best, convergence, stability, common):
        evidence: List[str] = []
        suggestions: List[str] = []
        warnings: List[str] = []

        if common["short_history"]:
            warnings.append(
                f"History has < {MIN_EPOCHS_FOR_STATS} epochs; "
                f"stability/convergence diagnostics are less reliable."
            )

        if common["history"]["nan_metrics"]:
            warnings.append(
                f"NaN values present in: {', '.join(common['history']['nan_metrics'])}"
            )

        # 1. Instability dominates everything
        if stability["unstable"]:
            evidence.append(
                f"val_loss std over last {RECENT_WINDOW} epochs = "
                f"{stability['val_loss_std_recent']:.4f} > {UNSTABLE_STD_THRESHOLD}"
            )
            if stability.get("grad_norm_warning"):
                evidence.append(
                    f"grad_norm_max={stability['grad_norm_max']:.2f} > {GRAD_NORM_WARN}"
                )
            suggestions += [
                "Reduce learning_rate (contrastive SSL is LR-sensitive)",
                "Increase warmup_steps",
                "Enable gradient_clip at 1.0",
                "Consider larger batch_size (more negatives per step)",
            ]
            return "UNSTABLE_VAL", evidence, suggestions, warnings

        # 2. Train/val divergence
        train_loss = self._recent(df, "train_loss")
        val_loss = self._recent(df, "val_loss")
        if len(train_loss) >= 3 and len(val_loss) >= 3:
            t_trend = train_loss[-1] - train_loss[0]
            v_trend = val_loss[-1] - val_loss[0]
            if t_trend < -0.05 and v_trend > 0.05:
                evidence.append(
                    f"train_loss {t_trend:+.4f} vs val_loss {v_trend:+.4f} "
                    f"over recent {RECENT_WINDOW} epochs"
                )
                suggestions += [
                    "Strengthen augmentation (SSL is overfitting)",
                    "Reduce capacity or add dropout in the projection head",
                    "Check whether val split shares recordings with train",
                ]
                return "OVERFIT", evidence, suggestions, warnings

        # 3. Plateau on val_loss
        if convergence["plateaued"]:
            evidence.append(
                f"No val_loss improvement for {convergence['epochs_since_best']} epochs "
                f"(best={best['value']:.4f} @ ep {best['epoch']})"
            )
            # Extra hint from contrastive loss magnitude
            if best.get("value") is not None and best["value"] < 2.0:
                suggestions.append(
                    "Consider raising contrastive temperature (e.g. 0.07 → 0.10) "
                    "to prevent degenerate low-entropy representations"
                )
            suggestions.append(
                "Consider stopping; validate representation via linear probe"
            )
            return "PLATEAU", evidence, suggestions, warnings

        if best.get("value") is not None:
            evidence.append(
                f"Best val_loss={best['value']:.4f} @ epoch {best['epoch']}; "
                f"no instability or divergence signals"
            )
        else:
            evidence.append("No val_loss available for diagnosis")
        suggestions.append(
            "Validate best checkpoint via frozen-encoder linear probe before "
            "comparing to other SimCLR runs"
        )
        return "HEALTHY", evidence, suggestions, warnings

    # ==================================================================
    # Generic (unknown objective)
    # ==================================================================
    def _analyze_generic(
        self, df: pd.DataFrame, test: Dict[str, Any], common: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Pick any val_* column
        val_cols = [
            c for c in df.columns if c.startswith("val_") and df[c].dtype.kind in "f"
        ]
        monitor = val_cols[0] if val_cols else None
        mode = "min" if monitor and "loss" in monitor else "max"
        best = (
            self._best_epoch(df, monitor, mode)
            if monitor
            else {"criterion": None, "mode": None, "epoch": None, "value": None}
        )
        return {
            "status": "ok",
            **common,
            "best": best,
            "convergence": (
                self._convergence(df, monitor, mode, best.get("epoch"))
                if monitor
                else {}
            ),
            "stability": self._stability(df),
            "test": test,
            "diagnosis": "UNKNOWN",
            "evidence": ["Objective could not be resolved; generic analysis only"],
            "suggestions": [
                "Add 'objective' to _build_logs so reports can apply "
                "objective-specific diagnosis rules"
            ],
            "warnings": ["Generic analysis has limited diagnostic value"],
        }

    # ==================================================================
    # Error report
    # ==================================================================
    def _error_report(self, message: str, status: str = "error") -> Dict[str, Any]:
        return {
            "run_id": self.run_dir.name,
            "status": status,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "error": message,
            "diagnosis": "UNKNOWN",
            "evidence": [],
            "suggestions": [],
            "warnings": [message],
        }

    # ==================================================================
    # Markdown rendering
    # ==================================================================
    def _render_markdown(self, report: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append(f"# Training Analysis — {report.get('run_id', '?')}")
        lines.append("")

        if report.get("status") != "ok":
            lines.append(f"**Status:** {report.get('status')}")
            lines.append(f"**Error:** {report.get('error', '—')}")
            return "\n".join(lines)

        # Header
        hist = report.get("history", {})
        best = report.get("best", {})
        lines.append(f"**Objective:** `{report.get('objective', '?')}`")
        lines.append(
            f"**Epochs:** {hist.get('first_epoch')}–{hist.get('last_epoch')} "
            f"({hist.get('n_epochs')} total, attempts: {hist.get('attempts', 0)})"
        )
        if best.get("value") is not None:
            lines.append(
                f"**Best {best['criterion']}:** {best['value']:.4f} "
                f"@ epoch {best['epoch']}"
            )
        lines.append(f"**Diagnosis:** **{report.get('diagnosis', 'UNKNOWN')}**")
        lines.append("")

        llm = report.get("llm")
        if llm and llm.get("conflicts_with_deterministic"):
            lines.append("> ⚠️ **LLM disagrees with deterministic diagnosis**")
            for c in llm["conflicts_with_deterministic"]:
                lines.append(
                    f"> - {c['deterministic_diagnosis']} vs "
                    f"{c['llm_diagnosis']}: {c['evidence_refs']}"
                )

        # Evidence
        if report.get("evidence"):
            lines.append("## Evidence")
            for e in report["evidence"]:
                lines.append(f"- {e}")
            lines.append("")

        # Test results
        test = report.get("test", {})
        if test.get("present"):
            lines.append("## Test Results")
            known = ["accuracy", "macro_f1", "weighted_f1"]
            rows = []
            for k in known:
                if k in test and isinstance(test[k], (int, float)):
                    rows.append([k, float(test[k])])
            if rows:
                lines.append(_md_table(["Metric", "Value"], rows))
            else:
                lines.append("_evaluation_metrics.json present but no known scalars_")
            lines.append("")

        # Overfit block (supervised only)
        overfit = report.get("overfit")
        if overfit:
            lines.append("## Overfit")
            lines.append(
                _md_table(
                    ["Field", "Value"],
                    [
                        ["train_acc (window)", overfit.get("train_acc")],
                        ["val_window_acc", overfit.get("val_window_acc")],
                        ["val_acc (recording)", overfit.get("val_recording_acc")],
                        ["gap (window vs window)", overfit.get("gap_window_vs_window")],
                        [
                            "gap (recording vs window)",
                            overfit.get("gap_recording_vs_window"),
                        ],
                        ["suspect", overfit.get("suspect")],
                    ],
                )
            )
            lines.append("")

        # Stability
        stab = report.get("stability", {})
        if stab:
            lines.append("## Stability")
            lines.append(
                _md_table(
                    ["Metric", "Value"],
                    [
                        ["val_loss std (recent)", stab.get("val_loss_std_recent")],
                        ["val_loss range (recent)", stab.get("val_loss_range_recent")],
                        ["grad_norm mean", stab.get("grad_norm_mean")],
                        ["grad_norm max", stab.get("grad_norm_max")],
                        ["grad_norm p95", stab.get("grad_norm_p95")],
                        ["unstable", stab.get("unstable")],
                    ],
                )
            )
            lines.append("")

        # Convergence
        conv = report.get("convergence", {})
        if conv:
            lines.append("## Convergence")
            lines.append(
                _md_table(
                    ["Field", "Value"],
                    [
                        ["epochs since best", conv.get("epochs_since_best")],
                        ["plateaued", conv.get("plateaued")],
                        ["improved recently", conv.get("improved_recently")],
                    ],
                )
            )
            lines.append("")

        # Suggestions
        if report.get("suggestions"):
            lines.append("## Suggestions")
            for s in report["suggestions"]:
                lines.append(f"- {s}")
            lines.append("")

        # Warnings
        if report.get("warnings"):
            lines.append("## Warnings")
            for w in report["warnings"]:
                lines.append(f"- ⚠️ {w}")
            lines.append("")

        return "\n".join(lines)


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if len(sys.argv) < 2:
        print("usage: python -m tools.mlops.log_analyst <run_dir>")
        sys.exit(2)
    TrainingLogAnalyst(Path(sys.argv[1])).analyze()
