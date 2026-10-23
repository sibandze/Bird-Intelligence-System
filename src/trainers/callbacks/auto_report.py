# src/trainers/callbacks/auto_report.py
import json
import logging

from .base import Callback, _atomic_write_text

logger = logging.getLogger(__name__)


class AutoReportCallback(Callback):
    def on_train_end(self, trainer):
        try:
            from tools.mlops.log_analyst import TrainingLogAnalyst

            report = TrainingLogAnalyst(trainer.run_dir).analyze()

            if trainer.config.get("reporting", {}).get("llm", {}).get("enabled", False):
                from tools.mlops.llm_analyst import enrich_report

                report = enrich_report(
                    report, trainer.config, trainer.run_dir / ".llm_cache"
                )
                _atomic_write_text(
                    trainer.run_dir / "report.json",
                    json.dumps(report, indent=2, allow_nan=False),
                )
                md = TrainingLogAnalyst(trainer.run_dir)._render_markdown(report)
                _atomic_write_text(trainer.run_dir / "report.md", md)
        except Exception as e:
            logger.warning(f"Auto-report failed: {e}", exc_info=True)
