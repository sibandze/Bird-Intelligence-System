# tools/mlops/hparam_advisor.py

"""
Cross-run hyperparameter advisor.

Reads report.json files from completed runs, groups them into comparable
cohorts (same objective + same monitoring criterion), and proposes the
next experiment's configuration.

Trust boundary
--------------
- Deterministic suggestions are produced by threshold rules over the cohort.
  They are the only suggestions applied to the output config.
- LLM suggestions come from report["llm"]["experiments"] (already tagged
  source="llm" by llm_analyst.py). They are ALWAYS requires_approval=true
  and are NEVER auto-applied. They are written to a `.provenance.json`
  sidecar for human review.
- The advisor never runs training, never promotes checkpoints, and never
  mutates base configs in place.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# Cohort-level thresholds (fractions of runs in a diagnosis class)
UNSTABLE_COHORT_FRACTION = 0.30
OVERFIT_COHORT_FRACTION = 0.40

# Augmentation / capacity bounds so suggestions don't drift off-cliff
AUG_PROB_MAX = 0.90
NOISE_LEVEL_MAX = 0.30
WEIGHT_DECAY_MAX_FACTOR = 4.0


# ── helpers ────────────────────────────────────────────────────────────


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into a copy of base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _expand_dotted(applied: Dict[str, Any]) -> Dict[str, Any]:
    """{'training.learning_rate': 5e-4} -> {'training': {'learning_rate': 5e-4}}"""
    out: Dict[str, Any] = {}
    for key, value in applied.items():
        parts = key.split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def _extract_best_value(report: Dict[str, Any]) -> Optional[float]:
    best = report.get("best") or {}
    val = best.get("value")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    compat = report.get("compat") or {}
    for key in ("best_val_acc", "best_val_loss"):
        if key in compat and compat[key] is not None:
            try:
                return float(compat[key])
            except (TypeError, ValueError):
                continue
    return None


def _validate_config(cfg: Dict[str, Any]) -> List[str]:
    """Return list of validation errors; empty list = config OK."""
    errors: List[str] = []
    training = cfg.get("training", {})

    lr = training.get("learning_rate")
    if lr is not None and (not isinstance(lr, (int, float)) or lr <= 0):
        errors.append(f"training.learning_rate must be > 0 (got {lr!r})")

    bs = training.get("batch_size")
    if bs is not None and (not isinstance(bs, int) or bs < 1):
        errors.append(f"training.batch_size must be >= 1 (got {bs!r})")

    ep = training.get("epochs")
    if ep is not None and (not isinstance(ep, int) or ep < 1):
        errors.append(f"training.epochs must be >= 1 (got {ep!r})")

    temp = training.get("temperature")
    if temp is not None:
        if not isinstance(temp, (int, float)) or not (0 < temp < 1):
            errors.append(f"training.temperature must be in (0, 1) (got {temp!r})")

    return errors


# ── advisor ────────────────────────────────────────────────────────────


class HyperparamAdvisor:
    def __init__(self, sweep_root: str = "runs"):
        self.root = Path(sweep_root)

    # ── loading ────────────────────────────────────────────────────────
    def _load_reports(self) -> List[Dict[str, Any]]:
        reports: List[Dict[str, Any]] = []
        if not self.root.exists():
            return reports
        for path in sorted(self.root.glob("*/report.json")):
            try:
                r = json.loads(path.read_text())
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")
                continue
            if r.get("status") != "ok":
                logger.info(f"Skipping {path.name}: status={r.get('status')}")
                continue
            reports.append(r)
        return reports

    def _load_base(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Base config not found: {path}")
        return yaml.safe_load(path.read_text()) or {}

    # ── cohorting ──────────────────────────────────────────────────────
    def _group_cohorts(
        self, reports: List[Dict[str, Any]]
    ) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        cohorts: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for r in reports:
            objective = r.get("objective", "unknown")
            criterion = (r.get("best") or {}).get("criterion") or "unknown"
            cohorts[(objective, criterion)].append(r)
        return dict(cohorts)

    # ── deterministic rules ────────────────────────────────────────────
    def _rule_suggestions(
        self,
        cohort: List[Dict[str, Any]],
        objective: str,
        criterion: str,
        base: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        n = len(cohort)
        diag_counts = Counter(r.get("diagnosis", "UNKNOWN") for r in cohort)
        suggestions: List[Dict[str, Any]] = []

        # ── Rule 1: unstable cohort → reduce LR + add grad clip ────────
        n_unstable = diag_counts.get("UNSTABLE_VAL", 0)
        if n > 0 and (n_unstable / n) >= UNSTABLE_COHORT_FRACTION:
            current_lr = base.get("training", {}).get("learning_rate", 1e-3)
            suggestions.append(
                {
                    "param": "training.learning_rate",
                    "from": current_lr,
                    "to": current_lr * 0.5,
                    "source": "deterministic",
                    "rationale": (
                        f"{n_unstable}/{n} runs diagnosed UNSTABLE_VAL "
                        f"({n_unstable / n:.0%})"
                    ),
                    "n_supporting_runs": n_unstable,
                }
            )
            if not base.get("training", {}).get("gradient_clip"):
                suggestions.append(
                    {
                        "param": "training.gradient_clip",
                        "from": None,
                        "to": 1.0,
                        "source": "deterministic",
                        "rationale": "Enable gradient clipping to stabilize training",
                        "n_supporting_runs": n_unstable,
                    }
                )

        # ── Rule 2: overfit cohort → stronger regularization ───────────
        n_overfit = diag_counts.get("OVERFIT", 0)
        if n > 0 and (n_overfit / n) >= OVERFIT_COHORT_FRACTION:
            if objective == "supervised_transformer":
                aug = base.get("augmentation", {})
                current_prob = aug.get("prob", 0.5)
                suggestions.append(
                    {
                        "param": "augmentation.prob",
                        "from": current_prob,
                        "to": min(current_prob + 0.2, AUG_PROB_MAX),
                        "source": "deterministic",
                        "rationale": (
                            f"{n_overfit}/{n} supervised runs diagnosed OVERFIT "
                            f"({n_overfit / n:.0%})"
                        ),
                        "n_supporting_runs": n_overfit,
                    }
                )
                current_wd = base.get("training", {}).get("weight_decay", 0.01)
                suggestions.append(
                    {
                        "param": "training.weight_decay",
                        "from": current_wd,
                        "to": current_wd * 2,
                        "source": "deterministic",
                        "rationale": "Increase weight decay to reduce overfitting",
                        "n_supporting_runs": n_overfit,
                    }
                )
            elif objective == "simclr":
                ac = base.get("acoustic_augmentation", {})
                current_noise = ac.get("noise_level", 0.05)
                suggestions.append(
                    {
                        "param": "acoustic_augmentation.noise_level",
                        "from": current_noise,
                        "to": min(current_noise * 2, NOISE_LEVEL_MAX),
                        "source": "deterministic",
                        "rationale": (
                            f"{n_overfit}/{n} SSL runs diagnosed OVERFIT "
                            f"({n_overfit / n:.0%})"
                        ),
                        "n_supporting_runs": n_overfit,
                    }
                )

        # ── Rule 3: task-too-easy → warn only, do not suggest changes ──
        n_easy = diag_counts.get("TASK_TOO_EASY_SUSPECTED", 0)
        if n > 0 and n_easy > 0:
            logger.warning(
                f"{n_easy}/{n} runs diagnosed TASK_TOO_EASY_SUSPECTED. "
                f"This usually points at data leakage or duplicate recordings, "
                f"not hyperparameters. Investigate data integrity before "
                f"changing training config."
            )

        return suggestions

    # ── LLM suggestions (never auto-applied) ───────────────────────────
    def _llm_suggestions(self, cohort: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in cohort:
            llm = r.get("llm") or {}
            if llm.get("error"):
                continue
            for exp in llm.get("experiments", []):
                if exp.get("source") != "llm":
                    continue
                out.append(
                    {
                        "param": exp.get("param"),
                        "from": exp.get("from"),
                        "to": exp.get("to"),
                        "source": "llm",
                        "rationale": exp.get("rationale"),
                        "expected_effect": exp.get("expected_effect"),
                        "from_run": r.get("run_id"),
                        "requires_approval": True,
                    }
                )
        return out

    # ── public API ─────────────────────────────────────────────────────
    def suggest_next(
        self,
        base_config_path: str = "configs/base.yaml",
        objective: Optional[str] = None,
        include_llm: bool = False,
        output_path: str = "configs/next_suggested.yaml",
    ) -> Optional[Dict[str, Any]]:
        reports = self._load_reports()
        if not reports:
            print(f"No valid report.json files under {self.root}/")
            print("  (run log_analyst on at least one completed run first)")
            return None

        cohorts = self._group_cohorts(reports)
        if objective:
            cohorts = {k: v for k, v in cohorts.items() if k[0] == objective}
            if not cohorts:
                print(f"No reports for objective={objective!r}")
                return None

        if len(cohorts) > 1:
            largest = max(cohorts.items(), key=lambda kv: len(kv[1]))
            print(
                f"WARNING: {len(cohorts)} cohorts found; using largest: "
                f"objective={largest[0][0]!r} criterion={largest[0][1]!r} "
                f"({len(largest[1])} runs)"
            )
            cohorts = {largest[0]: largest[1]}

        (objective, criterion), cohort = next(iter(cohorts.items()))

        base = self._load_base(Path(base_config_path))

        det_suggestions = self._rule_suggestions(cohort, objective, criterion, base)
        llm_suggestions = self._llm_suggestions(cohort) if include_llm else []

        if not det_suggestions and not llm_suggestions:
            print(f"Cohort {objective}/{criterion}: no actionable suggestions")
            self._print_cohort_summary(cohort, criterion)
            return None

        # Apply only deterministic suggestions
        applied = {s["param"]: s["to"] for s in det_suggestions}
        merged = _deep_merge(base, _expand_dotted(applied))

        errors = _validate_config(merged)
        if errors:
            print("REFUSED: merged config failed validation:")
            for e in errors:
                print(f"  - {e}")
            return None

        # Write output config
        out_cfg = Path(output_path)
        out_cfg.parent.mkdir(parents=True, exist_ok=True)
        out_cfg.write_text(yaml.dump(merged, sort_keys=False, default_flow_style=False))

        # Write provenance sidecar
        out_prov = out_cfg.with_suffix(".provenance.json")
        out_prov.write_text(
            json.dumps(
                {
                    "cohort": {
                        "objective": objective,
                        "criterion": criterion,
                        "n_runs": len(cohort),
                    },
                    "base_config": str(base_config_path),
                    "applied_deterministic": det_suggestions,
                    "not_applied_llm": llm_suggestions,
                },
                indent=2,
            )
        )

        # Summary
        self._print_cohort_summary(cohort, criterion)
        print(f"\n✓ Wrote {out_cfg}")
        print(f"✓ Wrote {out_prov}")
        if det_suggestions:
            print(f"\nApplied {len(det_suggestions)} deterministic suggestion(s):")
            for s in det_suggestions:
                print(f"  • {s['param']}: {s['from']} → {s['to']}  [{s['rationale']}]")
        if llm_suggestions:
            print(
                f"\n{len(llm_suggestions)} LLM suggestion(s) NOT applied "
                f"(requires approval). Review {out_prov}."
            )
        return merged

    def _print_cohort_summary(
        self, cohort: List[Dict[str, Any]], criterion: str
    ) -> None:
        values = [v for v in (_extract_best_value(r) for r in cohort) if v is not None]
        n = len(cohort)
        diag_counts = Counter(r.get("diagnosis", "UNKNOWN") for r in cohort)
        print(f"\nCohort: n={n}, criterion={criterion}")
        if values:
            print(
                f"  best {criterion}: "
                f"mean={statistics.mean(values):.4f}, "
                f"std={statistics.pstdev(values) if len(values) > 1 else 0:.4f}, "
                f"min={min(values):.4f}, max={max(values):.4f}"
            )
        print("  diagnoses:")
        for diag, cnt in diag_counts.most_common():
            print(f"    {diag}: {cnt}/{n} ({cnt / n:.0%})")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs", help="Sweep root (default: runs)")
    p.add_argument("--base", default="configs/base.yaml", help="Base config path")
    p.add_argument(
        "--objective", default=None, choices=[None, "supervised_transformer", "simclr"]
    )
    p.add_argument(
        "--include-llm",
        action="store_true",
        help="Include LLM suggestions (sidecar only; never applied)",
    )
    p.add_argument("--output", default="configs/next_suggested.yaml")
    args = p.parse_args()

    HyperparamAdvisor(sweep_root=args.root).suggest_next(
        base_config_path=args.base,
        objective=args.objective,
        include_llm=args.include_llm,
        output_path=args.output,
    )
