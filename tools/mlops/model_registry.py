# tools/mlops/model_registry.py
import shutil
import json
import torch
from pathlib import Path
from datetime import datetime


class ModelRegistry:
    """
    Minimal MLflow-like registry.
    Registry is just folder: models/registry/<name>-v<version>/
    """

    def __init__(self, registry_root="models/registry"):
        self.root = Path(registry_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def promote(self, run_dir: Path, model_name: str = "bird-encoder"):
        run_dir = Path(run_dir)
        best_ckpt = run_dir / "checkpoint_best.pth"
        report_path = run_dir / "report.json"
        label_map = run_dir / "label_mappings.json"

        if not best_ckpt.exists():
            print(f"No {best_ckpt}")
            return None

        # Find next version
        existing = list(self.root.glob(f"{model_name}-v*"))
        version = len(existing) + 1

        dest = self.root / f"{model_name}-v{version}"
        dest.mkdir()

        # Copy artifacts - this is the MLOps bundle
        shutil.copy(best_ckpt, dest / "model.pth")
        if label_map.exists():
            shutil.copy(label_map, dest / "label_mappings.json")
        if report_path.exists():
            shutil.copy(report_path, dest / "report.json")

        # Create model_card.json (governance)
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        card = {
            "model_name": model_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "source_run": str(run_dir),
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
