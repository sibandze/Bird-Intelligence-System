# tools/mlops/model_registry.py
import shutil
import json
import re
import torch
from pathlib import Path
from datetime import datetime

# Diagnoses that must never be promoted, even if a checkpoint exists.
BLOCKED_DIAGNOSES = {"UNSTABLE_VAL", "OVERFIT", "TASK_TOO_EASY_SUSPECTED", "UNKNOWN"}


class ModelRegistry:
    """
    Minimal MLflow-like registry.
    Registry is just folder: models/registry/<name>-v<version>/
    """

    def __init__(self, registry_root="models/registry"):
        self.root = Path(registry_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _next_version(self, model_name: str) -> int:
        # max + 1, not len + 1: deleting v2 must not let the next
        # promote reuse v2 and collide / overwrite.
        versions = []
        for p in self.root.glob(f"{model_name}-v*"):
            m = re.match(rf"^{re.escape(model_name)}-v(\d+)$", p.name)
            if m:
                versions.append(int(m.group(1)))
        return max(versions, default=0) + 1

    def promote(self, run_dir: Path, model_name: str = "bird-encoder"):
        run_dir = Path(run_dir)
        best_ckpt = run_dir / "checkpoint_best.pth"
        report_path = run_dir / "report.json"
        label_map = run_dir / "label_mappings.json"

        if not best_ckpt.exists():
            print(f"No {best_ckpt}")
            return None

        # ── Promotion gate: report must exist, be ok, and not blocked ──
        if not report_path.exists():
            print(f"Refusing to promote {run_dir.name}: no report.json")
            return None
        try:
            report = json.loads(report_path.read_text())
        except Exception as e:
            print(f"Refusing to promote {run_dir.name}: unreadable report.json: {e}")
            return None
        if report.get("status") != "ok":
            print(
                f"Refusing to promote {run_dir.name}: "
                f"report status={report.get('status')!r}"
            )
            return None
        diagnosis = report.get("diagnosis")
        if diagnosis in BLOCKED_DIAGNOSES:
            print(
                f"Refusing to promote {run_dir.name}: "
                f"diagnosis={diagnosis!r} is blocked"
            )
            return None

        version = self._next_version(model_name)
        dest = self.root / f"{model_name}-v{version}"
        dest.mkdir(exist_ok=False)

        # Copy artifacts - this is the MLOps bundle
        shutil.copy(best_ckpt, dest / "model.pth")
        if label_map.exists():
            shutil.copy(label_map, dest / "label_mappings.json")
        if report_path.exists():
            shutil.copy(report_path, dest / "report.json")

        # ── Safe checkpoint read ──
        # weights_only=True prevents arbitrary code execution from a
        # checkpoint. Fall back to False only for trusted local files
        # that fail the safe load (e.g. old checkpoints with custom objects).
        try:
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
        except Exception:
            print(
                f"Warning: safe load failed for {best_ckpt}, retrying with weights_only=False (trusted local file only)"
            )
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        # Checkpoint may be a raw state_dict, not a dict with metadata.
        if not isinstance(ckpt, dict):
            ckpt = {}

        # Create model_card.json (governance)
        card = {
            "model_name": model_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "source_run": str(run_dir),
            "source_diagnosis": diagnosis,
            "epoch": ckpt.get("epoch", -1),
            "trainer_state": ckpt.get("trainer_state", {}),
            "git_commit": ckpt.get("git_commit", None),
            "artifacts": ["model.pth", "label_mappings.json", "report.json"],
        }
        (dest / "model_card.json").write_text(json.dumps(card, indent=2))

        # Update latest pointer
        latest_link = self.root / f"{model_name}-latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(dest.name)

        print(f"✓ Promoted {run_dir.name} -> {dest} (v{version})")
        return dest


if __name__ == "__main__":
    import sys

    registry = ModelRegistry()
    registry.promote(
        run_dir=sys.argv[1],
        model_name=sys.argv[2] if len(sys.argv) > 2 else "bird-encoder",
    )
