# pipeline.py
# End-to-end pipeline runner for the block-window EVM cross-chain correlation project.

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .api_client import BlockchainClient
from .classify import (
    ClassificationConfig,
    run_classification_for_all_chains_and_windows,
)
from .config import AppConfig, load_config
from .features import build_features_for_all_windows
from .plots import (
    PlotConfig,
    render_plots_for_window,
    run_plot_stage_from_disk,
)
from .sampling import run_block_window_ingestion_for_all_chains
from .stats import StatsConfig, run_statistics_for_all_windows

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================


STAGE_INGEST = "ingest"
STAGE_CLASSIFY = "classify"
STAGE_FEATURES = "features"
STAGE_STATS = "stats"
STAGE_PLOTS = "plots"

ALL_STAGES = (
    STAGE_INGEST,
    STAGE_CLASSIFY,
    STAGE_FEATURES,
    STAGE_STATS,
    STAGE_PLOTS,
)


# ============================================================
# Data models
# ============================================================


@dataclass(frozen=True)
class PipelineStageStatus:
    stage_name: str
    started_at: str
    finished_at: Optional[str]
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineRunResult:
    run_id: str
    started_at: str
    finished_at: Optional[str]
    project_root: str
    enabled_chains: List[str]
    windows: List[int]
    stages_requested: List[str]
    stage_statuses: List[PipelineStageStatus] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "project_root": self.project_root,
            "enabled_chains": self.enabled_chains,
            "windows": self.windows,
            "stages_requested": self.stages_requested,
            "stage_statuses": [
                {
                    "stage_name": item.stage_name,
                    "started_at": item.started_at,
                    "finished_at": item.finished_at,
                    "success": item.success,
                    "details": item.details,
                }
                for item in self.stage_statuses
            ],
        }


# ============================================================
# Utilities
# ============================================================


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def utc_now_compact() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def normalize_stage_names(stage_names: Sequence[str]) -> List[str]:
    if not stage_names:
        return list(ALL_STAGES)

    normalized: List[str] = []
    for stage_name in stage_names:
        value = str(stage_name).strip().lower()
        if not value:
            continue
        if value not in ALL_STAGES:
            raise ValueError(
                f"Unknown stage {stage_name!r}. "
                f"Expected one of: {', '.join(ALL_STAGES)}"
            )
        if value not in normalized:
            normalized.append(value)

    if not normalized:
        return list(ALL_STAGES)

    return normalized


def setup_logging(log_dir: str | Path, level: int = logging.INFO) -> Path:
    log_dir = ensure_dir(log_dir)
    log_path = log_dir / f"pipeline_{utc_now_compact()}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logger.info("Logging initialized: %s", log_path)
    return log_path


def pipeline_manifest_path(app_config: AppConfig, run_id: str) -> Path:
    return app_config.paths.logs_dir / f"pipeline_run_{run_id}.json"


# ============================================================
# Plot stage runners
# ============================================================


def run_plot_stage_memory(
    app_config: AppConfig,
    *,
    classification_outputs: Dict[int, Dict[str, Any]],
    feature_outputs: Dict[int, Dict[str, Any]],
    stats_outputs: Dict[int, Dict[str, Any]],
    plot_config: Optional[PlotConfig] = None,
) -> Dict[int, Dict[str, str]]:
    """
    Render plots using in-memory outputs from classify/features/stats stages
    executed in the same pipeline session.
    """
    plot_config = plot_config or PlotConfig()
    outputs: Dict[int, Dict[str, str]] = {}

    for window_blocks in app_config.sampling.windows.block_counts:
        classification_for_window = classification_outputs.get(window_blocks, {})
        feature_for_window = feature_outputs.get(window_blocks, {})
        stats_for_window = stats_outputs.get(window_blocks, {})

        outputs[window_blocks] = render_plots_for_window(
            app_config=app_config,
            window_blocks=window_blocks,
            classification_outputs_for_window=classification_for_window,
            feature_outputs_for_window=feature_for_window,
            stats_outputs_for_window=stats_for_window,
            plot_config=plot_config,
        )

    return outputs


# ============================================================
# Pipeline stages
# ============================================================


def run_ingest_stage(
    app_config: AppConfig,
) -> Dict[str, Any]:
    with BlockchainClient(app_config.api) as client:
        outputs = run_block_window_ingestion_for_all_chains(
            client=client,
            app_config=app_config,
        )
    return outputs


def run_classify_stage(
    app_config: AppConfig,
    *,
    classification_config: Optional[ClassificationConfig] = None,
) -> Dict[int, Dict[str, Any]]:
    classification_config = classification_config or ClassificationConfig()
    return run_classification_for_all_chains_and_windows(
        app_config=app_config,
        config=classification_config,
        save_output=True,
    )


def run_features_stage(
    app_config: AppConfig,
) -> Dict[int, Dict[str, Any]]:
    return build_features_for_all_windows(
        app_config=app_config,
        save_output=True,
    )


def run_stats_stage(
    app_config: AppConfig,
    *,
    stats_config: Optional[StatsConfig] = None,
) -> Dict[int, Dict[str, Any]]:
    stats_config = stats_config or StatsConfig()
    return run_statistics_for_all_windows(
        app_config=app_config,
        config=stats_config,
        save_output=True,
    )


# ============================================================
# Main pipeline orchestration
# ============================================================


def execute_pipeline(
    *,
    project_root: Optional[str | Path] = None,
    stages: Optional[Sequence[str]] = None,
    classification_config: Optional[ClassificationConfig] = None,
    stats_config: Optional[StatsConfig] = None,
    plot_config: Optional[PlotConfig] = None,
) -> Dict[str, Any]:
    requested_stages = normalize_stage_names(stages)
    app_config = load_config(project_root=project_root)

    log_path = setup_logging(app_config.paths.logs_dir)
    logger.info("Pipeline requested stages: %s", requested_stages)
    logger.info("Enabled chains: %s", app_config.enabled_chain_names)
    logger.info("Window sizes (blocks): %s", list(app_config.sampling.windows.block_counts))
    logger.info("Project root: %s", app_config.paths.project_root)

    run_id = utc_now_compact()
    run_result = PipelineRunResult(
        run_id=run_id,
        started_at=utc_now_iso(),
        finished_at=None,
        project_root=str(app_config.paths.project_root),
        enabled_chains=list(app_config.enabled_chain_names),
        windows=list(app_config.sampling.windows.block_counts),
        stages_requested=list(requested_stages),
    )

    outputs: Dict[str, Any] = {
        "app_config": app_config,
        "log_path": str(log_path),
        "run_result": run_result,
    }

    classification_outputs: Dict[int, Dict[str, Any]] = {}
    feature_outputs: Dict[int, Dict[str, Any]] = {}
    stats_outputs: Dict[int, Dict[str, Any]] = {}
    plot_outputs: Dict[int, Dict[str, str]] = {}

    # --------------------------------------------------------
    # Stage runner helper
    # --------------------------------------------------------
    def run_stage(stage_name: str, fn, *args, **kwargs):
        started_at = utc_now_iso()
        logger.info("=== START STAGE: %s ===", stage_name)
        try:
            stage_output = fn(*args, **kwargs)
            finished_at = utc_now_iso()
            logger.info("=== FINISH STAGE: %s ===", stage_name)
            run_result.stage_statuses.append(
                PipelineStageStatus(
                    stage_name=stage_name,
                    started_at=started_at,
                    finished_at=finished_at,
                    success=True,
                    details={"output_type": type(stage_output).__name__},
                )
            )
            return stage_output
        except Exception as exc:
            finished_at = utc_now_iso()
            logger.exception("=== FAIL STAGE: %s ===", stage_name)
            run_result.stage_statuses.append(
                PipelineStageStatus(
                    stage_name=stage_name,
                    started_at=started_at,
                    finished_at=finished_at,
                    success=False,
                    details={"error": str(exc)},
                )
            )
            run_result.finished_at = finished_at
            write_json(pipeline_manifest_path(app_config, run_id), run_result.to_dict())
            raise

    # --------------------------------------------------------
    # Execute selected stages
    # --------------------------------------------------------
    if STAGE_INGEST in requested_stages:
        outputs["ingest"] = run_stage(
            STAGE_INGEST,
            run_ingest_stage,
            app_config,
        )

    if STAGE_CLASSIFY in requested_stages:
        classification_outputs = run_stage(
            STAGE_CLASSIFY,
            run_classify_stage,
            app_config,
            classification_config=classification_config,
        )
        outputs["classify"] = classification_outputs

    if STAGE_FEATURES in requested_stages:
        feature_outputs = run_stage(
            STAGE_FEATURES,
            run_features_stage,
            app_config,
        )
        outputs["features"] = feature_outputs

    if STAGE_STATS in requested_stages:
        stats_outputs = run_stage(
            STAGE_STATS,
            run_stats_stage,
            app_config,
            stats_config=stats_config,
        )
        outputs["stats"] = stats_outputs

    if STAGE_PLOTS in requested_stages:
        # Prefer in-memory outputs if all three are available in the same session.
        if classification_outputs and feature_outputs and stats_outputs:
            logger.info(
                "Plot stage will use in-memory outputs from classify/features/stats."
            )
            plot_outputs = run_stage(
                STAGE_PLOTS,
                run_plot_stage_memory,
                app_config,
                classification_outputs=classification_outputs,
                feature_outputs=feature_outputs,
                stats_outputs=stats_outputs,
                plot_config=plot_config,
            )
        else:
            # Fallback: load all plot inputs from disk.
            logger.info(
                "Plot stage is missing in-memory outputs; falling back to disk-backed plot loading."
            )
            plot_outputs = run_stage(
                STAGE_PLOTS,
                run_plot_stage_from_disk,
                app_config,
                plot_config=plot_config,
            )

        outputs["plots"] = plot_outputs

    run_result.finished_at = utc_now_iso()
    manifest_file = write_json(
        pipeline_manifest_path(app_config, run_id),
        run_result.to_dict(),
    )

    outputs["manifest_path"] = str(manifest_file)
    outputs["run_result"] = run_result
    return outputs


# ============================================================
# CLI
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the EVM cross-chain wallet correlation pipeline "
            "for exact block windows."
        )
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory. Defaults to current working directory.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=",".join(ALL_STAGES),
        help=(
            "Comma-separated pipeline stages. "
            f"Available: {', '.join(ALL_STAGES)}"
        ),
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations for permutation test in stats stage.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for statistical procedures.",
    )
    parser.add_argument(
        "--min-nonce-for-active",
        type=int,
        default=1,
        help="Minimum sender nonce to classify address as active.",
    )
    parser.add_argument(
        "--allow-unknown-eoa",
        action="store_true",
        help="Allow unknown EOA status during classification.",
    )
    parser.add_argument(
        "--skip-require-eoa",
        action="store_true",
        help="Do not require EOA status for presence/active/passive classification.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    stages = [part.strip() for part in args.stages.split(",") if part.strip()]

    classification_config = ClassificationConfig(
        min_nonce_for_active=args.min_nonce_for_active,
        require_eoa=not args.skip_require_eoa,
        allow_unknown_eoa=args.allow_unknown_eoa,
    )

    stats_config = StatsConfig(
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )

    outputs = execute_pipeline(
        project_root=args.project_root,
        stages=stages,
        classification_config=classification_config,
        stats_config=stats_config,
        plot_config=PlotConfig(),
    )

    manifest_path = outputs.get("manifest_path")
    print(f"Pipeline finished successfully. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()