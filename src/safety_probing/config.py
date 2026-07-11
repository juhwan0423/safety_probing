from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import load_json


@dataclass(frozen=True)
class LabelingConfig:
    schema_version: str
    observation_end_s: float
    roll_tip_threshold_deg: float
    support_loss_contact_count_lt: int
    support_loss_grace_steps: int
    terminal_window_s: float
    right_censor_min_abs_roll_deg: float
    right_censor_min_abs_roll_rate_deg_s: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LabelingConfig":
        return cls(
            schema_version=str(raw.get("schema_version", "censor-aware-lateral-tip-v3")),
            observation_end_s=float(
                raw.get("observation_end_s", 65 / 24)
            ),
            roll_tip_threshold_deg=float(raw.get("roll_tip_threshold_deg", 20.0)),
            support_loss_contact_count_lt=int(
                raw.get("support_loss_contact_count_lt", 3)
            ),
            support_loss_grace_steps=int(raw.get("support_loss_grace_steps", 6)),
            terminal_window_s=float(raw.get("terminal_window_s", 0.5)),
            right_censor_min_abs_roll_deg=float(
                raw.get("right_censor_min_abs_roll_deg", 0.5)
            ),
            right_censor_min_abs_roll_rate_deg_s=float(
                raw.get("right_censor_min_abs_roll_rate_deg_s", 2.0)
            ),
        )

    def validate(self) -> None:
        if self.observation_end_s < 0.0:
            raise ValueError("labeling.observation_end_s must be non-negative")
        if self.roll_tip_threshold_deg <= 0.0:
            raise ValueError("labeling.roll_tip_threshold_deg must be positive")
        if self.support_loss_contact_count_lt <= 0:
            raise ValueError("labeling.support_loss_contact_count_lt must be positive")
        if self.support_loss_grace_steps <= 0:
            raise ValueError("labeling.support_loss_grace_steps must be positive")
        if self.terminal_window_s <= 0.0:
            raise ValueError("labeling.terminal_window_s must be positive")
        if self.right_censor_min_abs_roll_deg < 0.0:
            raise ValueError(
                "labeling.right_censor_min_abs_roll_deg must be non-negative"
            )
        if self.right_censor_min_abs_roll_rate_deg_s < 0.0:
            raise ValueError(
                "labeling.right_censor_min_abs_roll_rate_deg_s must be non-negative"
            )


@dataclass(frozen=True)
class DatasetConfig:
    config_path: Path
    project_root: Path
    workspace_root: Path
    dataset_name: str
    dataset_version: str
    dataset_roots: tuple[Path, ...]
    source_config: Path
    reference_split_manifest: Path
    generated_split_manifest: Path
    output_dir: Path
    statistics_document: Path
    expected: dict[str, Any]
    labeling: LabelingConfig

    @classmethod
    def load(cls, path: str | Path) -> "DatasetConfig":
        config_path = Path(path).expanduser().resolve()
        raw = load_json(config_path)
        project_root = config_path.parent.parent
        workspace_root = (config_path.parent / raw.get("workspace_root", "../..")).resolve()

        def resolve_workspace(value: str) -> Path:
            candidate = Path(value).expanduser()
            return candidate.resolve() if candidate.is_absolute() else (workspace_root / candidate).resolve()

        def resolve_project(value: str) -> Path:
            candidate = Path(value).expanduser()
            return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()

        config = cls(
            config_path=config_path,
            project_root=project_root,
            workspace_root=workspace_root,
            dataset_name=str(raw["dataset_name"]),
            dataset_version=str(raw["dataset_version"]),
            dataset_roots=tuple(resolve_workspace(value) for value in raw["dataset_roots"]),
            source_config=resolve_workspace(raw["source_config"]),
            reference_split_manifest=resolve_workspace(raw["reference_split_manifest"]),
            generated_split_manifest=resolve_project(raw["generated_split_manifest"]),
            output_dir=resolve_project(raw.get("output_dir", "artifacts/current")),
            statistics_document=resolve_project(
                raw.get("statistics_document", "docs/DATASET_STATISTICS_KO.md")
            ),
            expected=dict(raw["expected"]),
            labeling=LabelingConfig.from_dict(dict(raw.get("labeling", {}))),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.dataset_roots:
            raise ValueError("dataset_roots must contain at least one path")
        if not self.source_config.is_file():
            raise FileNotFoundError(f"Source experiment config not found: {self.source_config}")
        if not self.reference_split_manifest.is_file():
            raise FileNotFoundError(
                f"Reference split manifest not found: {self.reference_split_manifest}"
            )
        for key in ("pair_count", "run_count", "physics_rows_per_run", "rgb_frames_per_camera_per_run"):
            if int(self.expected[key]) <= 0:
                raise ValueError(f"expected.{key} must be positive")
        self.labeling.validate()
