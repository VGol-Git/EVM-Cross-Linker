# pipeline.py
# End-to-end pipeline runner for the block-window EVM cross-chain correlation project.

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .api_client import BlockchainClient
from .classify import (
    ClassificationConfig,
    run_classification_for_all_chains_and_windows,
)
from .config import AppConfig, load_config
from .features import build_features_for_all_windows
from .plots import (
    PlotConfig,
    build_chain_status_count_table,
    plot_chain_status_counts,
    plot_first_activity_delta_histogram,
    plot_pairwise_correlation_heatmap,
    plot_pairwise_intersection_heatmap,
    plot_pairwise_jaccard_heatmap,
    plot_summary_table,
    plot_value_sent_scatter,
    save_figure,
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


def stage_output_dir(
    app_config: AppConfig,
    stage_name: str,
    window_blocks: Optional[int] = None,
) -> Path:
    base = ensure_dir(app_config.paths.outputs_dir / "pipeline")
    if window_blocks is None:
        return ensure_dir(base / stage_name)
    return ensure_dir(base / stage_name / f"window_{window_blocks}")


# ============================================================
# Plot helpers
# ============================================================


def _filter_feature_stats_for_metric(
    feature_stats_df: pd.DataFrame,
    *,
    feature_name: str,
    metric_col: str,
) -> pd.DataFrame:
    if feature_stats_df.empty:
        return pd.DataFrame()

    required = {"chain_a", "chain_b", "feature_a", "feature_b", metric_col}
    missing = required - set(feature_stats_df.columns)
    if missing:
        raise ValueError(
            f"feature_stats_df missing required columns: {sorted(missing)}"
        )

    df = feature_stats_df.copy()
    df = df[
        (df["feature_a"].astype(str) == feature_name)
        & (df["feature_b"].astype(str) == feature_name)
    ].copy()

    if df.empty:
        return df

    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    return df


def run_plots_for_window(
    app_config: AppConfig,
    *,
    window_blocks: int,
    classification_outputs_for_window: Dict[str, Any],
    feature_outputs_for_window: Dict[str, Any],
    stats_outputs_for_window: Dict[str, Any],
    plot_config: Optional[PlotConfig] = None,
) -> Dict[str, str]:
    plot_config = plot_config or PlotConfig()
    saved_paths: Dict[str, str] = {}

    output_dir = stage_output_dir(
        app_config=app_config,
        stage_name=STAGE_PLOTS,
        window_blocks=window_blocks,
    )

    # --------------------------------------------------------
    # 1. Chain status counts
    # --------------------------------------------------------
    status_tables_by_chain = classification_outputs_for_window["status_tables"]
    chain_counts_df = build_chain_status_count_table(status_tables_by_chain)

    fig, _ = plot_chain_status_counts(
        chain_counts_df,
        title=f"Wallet counts by chain (window={window_blocks} blocks)",
        config=plot_config,
    )
    path = save_figure(fig, output_dir / "chain_status_counts.png", plot_config)
    saved_paths["chain_status_counts"] = str(path)

    # --------------------------------------------------------
    # 2. Overlap / Jaccard heatmaps
    # --------------------------------------------------------
    matching = classification_outputs_for_window["matching"]

    pairwise_plot_specs = [
        ("present_pairwise", "present_intersection_heatmap", "Present overlap count"),
        ("active_pairwise", "active_intersection_heatmap", "Active overlap count"),
        ("passive_pairwise", "passive_intersection_heatmap", "Passive overlap count"),
    ]

    for key, out_name, title in pairwise_plot_specs:
        df = matching.get(key, pd.DataFrame())
        if df is not None and not df.empty:
            fig, _ = plot_pairwise_intersection_heatmap(
                df,
                title=f"{title} (window={window_blocks} blocks)",
                config=plot_config,
            )
            path = save_figure(fig, output_dir / f"{out_name}.png", plot_config)
            saved_paths[out_name] = str(path)

    jaccard_plot_specs = [
        ("present_pairwise", "present_jaccard_heatmap", "Present Jaccard similarity"),
        ("active_pairwise", "active_jaccard_heatmap", "Active Jaccard similarity"),
        ("passive_pairwise", "passive_jaccard_heatmap", "Passive Jaccard similarity"),
        (
            "mixed_active_passive",
            "mixed_active_passive_jaccard_heatmap",
            "Active vs Passive Jaccard similarity",
        ),
    ]

    for key, out_name, title in jaccard_plot_specs:
        df = matching.get(key, pd.DataFrame())
        if df is not None and not df.empty:
            fig, _ = plot_pairwise_jaccard_heatmap(
                df,
                title=f"{title} (window={window_blocks} blocks)",
                config=plot_config,
            )
            path = save_figure(fig, output_dir / f"{out_name}.png", plot_config)
            saved_paths[out_name] = str(path)

    # --------------------------------------------------------
    # 3. Feature-alignment plots per chain pair
    # --------------------------------------------------------
    pairwise_alignments = feature_outputs_for_window["pairwise_alignments"]

    for (chain_a, chain_b), alignment_df in pairwise_alignments.items():
        if alignment_df is None or alignment_df.empty:
            continue

        pair_prefix = f"{chain_a}_vs_{chain_b}"

        try:
            fig, _ = plot_value_sent_scatter(
                alignment_df,
                chain_a=chain_a,
                chain_b=chain_b,
                title=f"Sent value: {chain_a} vs {chain_b} ({window_blocks} blocks)",
                config=plot_config,
            )
            path = save_figure(
                fig,
                output_dir / f"{pair_prefix}_value_sent_scatter.png",
                plot_config,
            )
            saved_paths[f"{pair_prefix}_value_sent_scatter"] = str(path)
        except ValueError as exc:
            logger.warning(
                "[window=%s] Skip value scatter for %s vs %s: %s",
                window_blocks,
                chain_a,
                chain_b,
                exc,
            )

        try:
            fig, _ = plot_first_activity_delta_histogram(
                alignment_df,
                chain_a=chain_a,
                chain_b=chain_b,
                title=f"Δ first activity: {chain_a} vs {chain_b} ({window_blocks} blocks)",
                config=plot_config,
            )
            path = save_figure(
                fig,
                output_dir / f"{pair_prefix}_first_activity_delta_hist.png",
                plot_config,
            )
            saved_paths[f"{pair_prefix}_first_activity_delta_hist"] = str(path)
        except ValueError as exc:
            logger.warning(
                "[window=%s] Skip first-activity histogram for %s vs %s: %s",
                window_blocks,
                chain_a,
                chain_b,
                exc,
            )

    # --------------------------------------------------------
    # 4. Correlation heatmaps from stats layer
    # --------------------------------------------------------
    feature_stats_df = stats_outputs_for_window["features"].get(
        "feature_stats",
        pd.DataFrame(),
    )

    correlation_specs = [
        ("value_sent_wei", "pearson_r", "value_sent_pearson_heatmap", "Pearson r: sent value"),
        ("value_sent_wei", "spearman_r", "value_sent_spearman_heatmap", "Spearman r: sent value"),
        ("tx_frequency_per_day", "pearson_r", "tx_frequency_pearson_heatmap", "Pearson r: tx/day"),
        ("tx_frequency_per_day", "spearman_r", "tx_frequency_spearman_heatmap", "Spearman r: tx/day"),
        ("unique_counterparties", "pearson_r", "counterparties_pearson_heatmap", "Pearson r: counterparties"),
        ("unique_counterparties", "spearman_r", "counterparties_spearman_heatmap", "Spearman r: counterparties"),
    ]

    for feature_name, metric_col, out_name, title in correlation_specs:
        df = _filter_feature_stats_for_metric(
            feature_stats_df,
            feature_name=feature_name,
            metric_col=metric_col,
        )
        if df.empty:
            continue

        try:
            fig, _ = plot_pairwise_correlation_heatmap(
                df,
                metric_col=metric_col,
                title=f"{title} (window={window_blocks} blocks)",
                config=plot_config,
            )
            path = save_figure(fig, output_dir / f"{out_name}.png", plot_config)
            saved_paths[out_name] = str(path)
        except ValueError as exc:
            logger.warning(
                "[window=%s] Skip correlation heatmap %s: %s",
                window_blocks,
                out_name,
                exc,
            )

    # --------------------------------------------------------
    # 5. Summary table plot
    # --------------------------------------------------------
    summary_df = stats_outputs_for_window.get("summary", pd.DataFrame())
    if summary_df is not None and not summary_df.empty:
        fig, _ = plot_summary_table(
            summary_df,
            title=f"Statistical summary (window={window_blocks} blocks)",
            max_rows=20,
            config=plot_config,
        )
        path = save_figure(fig, output_dir / "stats_summary_table.png", plot_config)
        saved_paths["stats_summary_table"] = str(path)

    return saved_paths


def run_plot_stage(
    app_config: AppConfig,
    *,
    classification_outputs: Dict[int, Dict[str, Any]],
    feature_outputs: Dict[int, Dict[str, Any]],
    stats_outputs: Dict[int, Dict[str, Any]],
    plot_config: Optional[PlotConfig] = None,
) -> Dict[int, Dict[str, str]]:
    plot_config = plot_config or PlotConfig()

    outputs: Dict[int, Dict[str, str]] = {}
    for window_blocks in app_config.sampling.windows.block_counts:
        classification_for_window = classification_outputs.get(window_blocks, {})
        feature_for_window = feature_outputs.get(window_blocks, {})
        stats_for_window = stats_outputs.get(window_blocks, {})

        outputs[window_blocks] = run_plots_for_window(
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
        if not classification_outputs:
            raise RuntimeError(
                "Plot stage requires classification outputs. "
                "Run classify stage in the same pipeline execution."
            )
        if not feature_outputs:
            raise RuntimeError(
                "Plot stage requires feature outputs. "
                "Run features stage in the same pipeline execution."
            )
        if not stats_outputs:
            raise RuntimeError(
                "Plot stage requires stats outputs. "
                "Run stats stage in the same pipeline execution."
            )

        plot_outputs = run_stage(
            STAGE_PLOTS,
            run_plot_stage,
            app_config,
            classification_outputs=classification_outputs,
            feature_outputs=feature_outputs,
            stats_outputs=stats_outputs,
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