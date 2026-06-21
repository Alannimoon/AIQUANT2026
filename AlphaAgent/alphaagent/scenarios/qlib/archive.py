from __future__ import annotations

import os
import random
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from alphaagent.core.proposal import Trace

if TYPE_CHECKING:
    from alphaagent.components.coder.factor_coder.factor import FactorTask


def _read_default_complexity_metric() -> str:
    env_key = "QLIB_FACTOR_ARCHIVE_COMPLEXITY_METRIC"
    value = os.getenv(env_key)
    if value is not None:
        return _clean_env_value(value) or "depth"

    for env_path in _iter_env_files():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, raw_value = stripped.split("=", 1)
                if key.strip() == env_key:
                    return _clean_env_value(raw_value) or "depth"
        except OSError:
            continue
    return "depth"


def _iter_env_files():
    seen = set()
    roots = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    for root in roots:
        env_path = root / ".env"
        if env_path in seen:
            continue
        seen.add(env_path)
        if env_path.exists():
            yield env_path


def _clean_env_value(value: str) -> str:
    value = value.strip()
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    return value.strip("\"'")


def _read_int_env(env_key: str, default: int) -> int:
    value = os.getenv(env_key)
    if value is not None:
        try:
            return int(_clean_env_value(value))
        except ValueError:
            return default

    for env_path in _iter_env_files():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, raw_value = stripped.split("=", 1)
                if key.strip() != env_key:
                    continue
                try:
                    return int(_clean_env_value(raw_value))
                except ValueError:
                    return default
        except OSError:
            continue
    return default


def _read_float_env(env_key: str, default: float) -> float:
    value = os.getenv(env_key)
    if value is not None:
        try:
            return float(_clean_env_value(value))
        except ValueError:
            return default

    for env_path in _iter_env_files():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, raw_value = stripped.split("=", 1)
                if key.strip() != env_key:
                    continue
                try:
                    return float(_clean_env_value(raw_value))
                except ValueError:
                    return default
        except OSError:
            continue
    return default


DEFAULT_FACTOR_CATEGORIES: tuple[str, ...] = (
    "momentum",
    "reversal",
    "volatility",
    "volume-price",
    "cross-sectional",
)

DEFAULT_DEPTH_BINS: tuple[int, ...] = (1, 2, 3, 4, 5)
DEFAULT_COMPLEXITY_METRIC = _read_default_complexity_metric()
SUPPORTED_COMPLEXITY_METRICS: tuple[str, ...] = ("depth", "vertex")
DEFAULT_VERTEX_COUNT_THRESHOLDS: tuple[int, ...] = (3, 6, 10, 15, 20)
DEFAULT_QUALITY_METRIC = "Rank IC"
DEFAULT_ARCHIVE_DUPLICATION_THRESHOLD = _read_int_env("QLIB_FACTOR_ARCHIVE_DUPLICATION_THRESHOLD", 8)
DEFAULT_ARCHIVE_CORR_REG_COEF = _read_float_env("QLIB_FACTOR_ARCHIVE_CORR_REG_COEF", 0.08)
DEFAULT_ARCHIVE_CORR_REG_THRESHOLD = _read_float_env("QLIB_FACTOR_ARCHIVE_CORR_REG_THRESHOLD", 0.30)
DEFAULT_ARCHIVE_CORR_REG_TOP_K = max(1, _read_int_env("QLIB_FACTOR_ARCHIVE_CORR_REG_TOP_K", 3))
_RANK_IC_QUALITY_KEYS: tuple[str, ...] = ("Rank IC", "RankIC", "rank_ic", "rank ic", "rank-ic")
_TRAIN_RANK_IC_QUALITY_KEYS: tuple[str, ...] = (
    "train_Rank IC",
    "train_RankIC",
    "train_rank_ic",
    "train rank ic",
    "train_rank-ic",
)
_IC_QUALITY_KEYS: tuple[str, ...] = ("IC", "ic")
_QUALITY_KEY_PRIORITY: tuple[str, ...] = _TRAIN_RANK_IC_QUALITY_KEYS + _RANK_IC_QUALITY_KEYS + _IC_QUALITY_KEYS
_ARCHIVE_FACTOR_VALUE_CACHE: dict[str, Any] = {}


def register_archive_factor_values(task, values, values_path: str | Path | None = None) -> None:
    """Register disk-backed factor values for archive corr checks."""
    key = str(Path(values_path).expanduser()) if values_path is not None else f"{id(task)}:{getattr(task, 'factor_name', '')}"
    old_key = getattr(task, "archive_factor_values_key", None)
    if old_key and old_key != key:
        _ARCHIVE_FACTOR_VALUE_CACHE.pop(old_key, None)
    _ARCHIVE_FACTOR_VALUE_CACHE[key] = values
    setattr(task, "archive_factor_values_key", key)
    if hasattr(task, "archive_factor_values"):
        delattr(task, "archive_factor_values")
    if values_path is not None:
        setattr(task, "archive_factor_values_path", str(values_path))
    try:
        setattr(task, "archive_factor_values_count", int(len(values)))
    except TypeError:
        setattr(task, "archive_factor_values_count", None)


def evict_archive_factor_values(task) -> None:
    key = getattr(task, "archive_factor_values_key", None)
    if key:
        _ARCHIVE_FACTOR_VALUE_CACHE.pop(key, None)
        delattr(task, "archive_factor_values_key")
    if hasattr(task, "archive_factor_values"):
        delattr(task, "archive_factor_values")


class EliteAlphaTrace(Trace):
    """Trace with a MAP-Elites archive alongside the linear history."""

    def __init__(
        self,
        scen,
        knowledge_base=None,
        archive: EliteArchive | None = None,
        archive_complexity_metric: str = DEFAULT_COMPLEXITY_METRIC,
        archive_vertex_count_thresholds: Sequence[int] | None = None,
    ) -> None:
        super().__init__(scen=scen, knowledge_base=knowledge_base)
        archive_kwargs = {"complexity_metric": archive_complexity_metric}
        if archive_vertex_count_thresholds is not None:
            archive_kwargs["vertex_count_thresholds"] = archive_vertex_count_thresholds
        self.archive = archive or EliteArchive(**archive_kwargs)


@dataclass(frozen=True, slots=True)
class BehaviorDescriptor:
    """Cell coordinate in the MAP-Elites archive."""

    category: str
    depth_bin: int

    def key(self) -> tuple[str, int]:
        return self.category, self.depth_bin


@dataclass(slots=True)
class EliteRecord:
    """A FactorTask plus the MAP-Elites information needed to place it."""

    task: FactorTask
    descriptor: BehaviorDescriptor
    quality: float

    @property
    def category(self) -> str:
        return self.descriptor.category

    @property
    def depth_bin(self) -> int:
        return self.descriptor.depth_bin

    @property
    def factor_name(self) -> str:
        return self.task.factor_name

    @property
    def factor_expression(self) -> str | None:
        return self.task.factor_expression

    @property
    def factor_description(self) -> str:
        return self.task.factor_description

    @property
    def factor_formulation(self) -> str:
        return self.task.factor_formulation

    @property
    def variables(self) -> dict[str, Any]:
        return self.task.variables

    @property
    def factor_implementation(self) -> bool:
        return self.task.factor_implementation

    @property
    def factor_ast_depth(self) -> int | None:
        return getattr(self.task, "factor_ast_depth", None)

    @property
    def factor_ast_node_count(self) -> int | None:
        return getattr(self.task, "factor_ast_node_count", None)

    @property
    def factor_complexity_metric(self) -> str | None:
        return getattr(self.task, "factor_complexity_metric", None)

    @property
    def factor_complexity_value(self) -> int | None:
        return getattr(self.task, "factor_complexity_value", None)

    @property
    def quality_segment(self) -> str | None:
        return getattr(self.task, "archive_quality_segment", None)

    @property
    def train_rank_ic(self) -> float | None:
        return getattr(self.task, "train_rank_ic", None)

    @property
    def train_rank_icir(self) -> float | None:
        return getattr(self.task, "train_rank_icir", None)

    @property
    def validation_rank_ic(self) -> float | None:
        return getattr(self.task, "validation_rank_ic", None)

    @property
    def validation_rank_icir(self) -> float | None:
        return getattr(self.task, "validation_rank_icir", None)

    @property
    def validation_raw_rank_ic(self) -> float | None:
        return getattr(self.task, "validation_raw_rank_ic", None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "factor_expression": self.factor_expression,
            "factor_description": self.factor_description,
            "factor_formulation": self.factor_formulation,
            "variables": self.variables,
            "factor_implementation": self.factor_implementation,
            "category": self.category,
            "depth_bin": self.depth_bin,
            "factor_ast_depth": self.factor_ast_depth,
            "factor_ast_node_count": self.factor_ast_node_count,
            "factor_complexity_metric": self.factor_complexity_metric,
            "factor_complexity_value": self.factor_complexity_value,
            "quality_metric": DEFAULT_QUALITY_METRIC,
            "quality_segment": self.quality_segment,
            "quality": self.quality,
            "train_Rank IC": self.train_rank_ic,
            "train_Rank ICIR": self.train_rank_icir,
            "validation_Rank IC": self.validation_rank_ic,
            "validation_Rank ICIR": self.validation_rank_icir,
            "validation_raw_Rank IC": self.validation_raw_rank_ic,
        }

    @classmethod
    def from_task(
        cls,
        task: FactorTask,
        *,
        descriptor: BehaviorDescriptor,
        quality: float,
    ) -> EliteRecord:
        return cls(task=task, descriptor=descriptor, quality=quality)


@dataclass(slots=True)
class EliteArchiveHistory:
    """One attempted archive update."""

    record: EliteRecord
    incumbent: EliteRecord | None
    accepted: bool
    rejection_reason: str | None = None
    regularized_quality: float | None = None
    corr_penalty: float | None = None
    corr_reg_coef: float | None = None
    corr_reg_threshold: float | None = None
    corr_top_k: int | None = None
    top_k_avg_cross_category_corr: float | None = None
    max_cross_category_corr: float | None = None
    corr_match_factor_name: str | None = None
    corr_match_category: str | None = None
    corr_match_depth_bin: int | None = None
    corr_compared_count: int | None = None
    incumbent_regularized_quality: float | None = None
    incumbent_corr_penalty: float | None = None
    incumbent_max_cross_category_corr: float | None = None
    incumbent_corr_match_factor_name: str | None = None
    incumbent_corr_compared_count: int | None = None

    @property
    def descriptor(self) -> BehaviorDescriptor:
        return self.record.descriptor

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_name": self.record.factor_name,
            "factor_expression": self.record.factor_expression,
            "category": self.record.category,
            "depth_bin": self.record.depth_bin,
            "factor_ast_depth": self.record.factor_ast_depth,
            "factor_ast_node_count": self.record.factor_ast_node_count,
            "factor_complexity_metric": self.record.factor_complexity_metric,
            "factor_complexity_value": self.record.factor_complexity_value,
            "quality_metric": DEFAULT_QUALITY_METRIC,
            "quality_segment": self.record.quality_segment,
            "quality": self.record.quality,
            "train_Rank IC": self.record.train_rank_ic,
            "train_Rank ICIR": self.record.train_rank_icir,
            "validation_Rank IC": self.record.validation_rank_ic,
            "validation_Rank ICIR": self.record.validation_rank_icir,
            "validation_raw_Rank IC": self.record.validation_raw_rank_ic,
            "accepted": self.accepted,
            "incumbent_factor_name": self.incumbent.factor_name if self.incumbent else None,
            "incumbent_quality": self.incumbent.quality if self.incumbent else None,
            "rejection_reason": getattr(self, "rejection_reason", None),
            "regularized_quality": getattr(self, "regularized_quality", None),
            "corr_penalty": getattr(self, "corr_penalty", None),
            "corr_reg_coef": getattr(self, "corr_reg_coef", None),
            "corr_reg_threshold": getattr(self, "corr_reg_threshold", None),
            "corr_top_k": getattr(self, "corr_top_k", None),
            "top_k_avg_cross_category_corr": getattr(self, "top_k_avg_cross_category_corr", None),
            "max_cross_category_corr": getattr(self, "max_cross_category_corr", None),
            "corr_match_factor_name": getattr(self, "corr_match_factor_name", None),
            "corr_match_category": getattr(self, "corr_match_category", None),
            "corr_match_depth_bin": getattr(self, "corr_match_depth_bin", None),
            "corr_compared_count": getattr(self, "corr_compared_count", None),
            "incumbent_regularized_quality": getattr(self, "incumbent_regularized_quality", None),
            "incumbent_corr_penalty": getattr(self, "incumbent_corr_penalty", None),
            "incumbent_max_cross_category_corr": getattr(self, "incumbent_max_cross_category_corr", None),
            "incumbent_corr_match_factor_name": getattr(self, "incumbent_corr_match_factor_name", None),
            "incumbent_corr_compared_count": getattr(self, "incumbent_corr_compared_count", None),
        }


@dataclass(frozen=True, slots=True)
class ArchiveCorrRegularization:
    """Cross-category factor correlation penalty for archive replacement."""

    penalty: float = 0.0
    corr_reg_coef: float = DEFAULT_ARCHIVE_CORR_REG_COEF
    corr_reg_threshold: float = DEFAULT_ARCHIVE_CORR_REG_THRESHOLD
    corr_top_k: int = DEFAULT_ARCHIVE_CORR_REG_TOP_K
    top_k_avg_abs_corr: float | None = None
    max_abs_corr: float | None = None
    matched_factor_name: str | None = None
    matched_category: str | None = None
    matched_depth_bin: int | None = None
    compared_count: int = 0
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveSimilarityMatch:
    """Largest cross-category AST overlap found against an archive record."""

    record: EliteRecord
    duplicated_subtree_size: int
    duplicated_subtree: str | None
    proposed_node_count: int

    @property
    def should_reject(self) -> bool:
        return (
            self.duplicated_subtree_size > DEFAULT_ARCHIVE_DUPLICATION_THRESHOLD
            or self.duplicated_subtree_size >= self.proposed_node_count
        )

    def rejection_reason(self) -> str:
        return (
            "cross-category AST similarity with "
            f"{self.record.factor_name} in cell=({self.record.category}, {self.record.depth_bin}); "
            f"duplicated_subtree_size={self.duplicated_subtree_size}, "
            f"proposed_node_count={self.proposed_node_count}, "
            f"threshold={DEFAULT_ARCHIVE_DUPLICATION_THRESHOLD}, "
            f"duplicated_subtree={self.duplicated_subtree}"
        )


class EliteArchive:
    """
    MAP-Elites archive for factor mining.

    Each cell is indexed by (factor category, complexity bin), and keeps only
    the highest-quality factor observed for that cell.
    """

    def __init__(
        self,
        categories: Sequence[str] = DEFAULT_FACTOR_CATEGORIES,
        depth_bins: Sequence[int] = DEFAULT_DEPTH_BINS,
        complexity_metric: str = DEFAULT_COMPLEXITY_METRIC,
        vertex_count_thresholds: Sequence[int] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.categories: tuple[str, ...] = tuple(categories)
        self.depth_bins: tuple[int, ...] = tuple(depth_bins)
        self.complexity_metric = self._normalize_complexity_metric(complexity_metric)
        if self.complexity_metric == "vertex":
            thresholds = DEFAULT_VERTEX_COUNT_THRESHOLDS if vertex_count_thresholds is None else vertex_count_thresholds
            self.vertex_count_thresholds: tuple[int, ...] | None = tuple(int(v) for v in thresholds)
        else:
            self.vertex_count_thresholds = None
        if len(self.depth_bins) != 5:
            raise ValueError("EliteArchive currently expects exactly five complexity bins.")
        if self.vertex_count_thresholds is not None and len(self.vertex_count_thresholds) not in {
            len(self.depth_bins) - 1,
            len(self.depth_bins),
        }:
            raise ValueError(
                "vertex_count_thresholds must have either one fewer value than bins "
                "or one upper-bound value per bin."
            )
        self._cells: dict[BehaviorDescriptor, EliteRecord] = {}
        self.hist: list[EliteArchiveHistory] = []
        self._rng = rng or random.Random()

    def __len__(self) -> int:
        return len(self._cells)

    def __contains__(self, descriptor: BehaviorDescriptor | tuple[str, int]) -> bool:
        return self.get_descriptor(descriptor) in self._cells

    def __iter__(self):
        return iter(self._cells.values())

    @property
    def cells(self) -> Mapping[BehaviorDescriptor, EliteRecord]:
        return self._cells

    @property
    def total_cells(self) -> int:
        return len(self.categories) * len(self.depth_bins)

    def get_descriptor(self, descriptor: BehaviorDescriptor | tuple[str, int]) -> BehaviorDescriptor:
        if isinstance(descriptor, BehaviorDescriptor):
            return descriptor
        category, depth_bin = descriptor
        return BehaviorDescriptor(category=category, depth_bin=int(depth_bin))

    def normalize_category(self, category: str) -> str:
        category = category.strip().lower().replace("_", "-")
        aliases = {
            "volume": "volume-price",
            "volume price": "volume-price",
            "volume-price": "volume-price",
            "cross sectional": "cross-sectional",
            "cross-sectional": "cross-sectional",
            "cross_sectional": "cross-sectional",
        }
        return aliases.get(category, category)

    def make_descriptor(self, category: str, complexity_value: int) -> BehaviorDescriptor:
        category = self.normalize_category(category)
        if category not in self.categories:
            raise ValueError(f"Unknown factor category: {category!r}. Expected one of {self.categories}.")
        return BehaviorDescriptor(category=category, depth_bin=self.complexity_to_bin(complexity_value))

    def complexity_to_bin(self, complexity_value: int) -> int:
        if self.complexity_metric == "depth":
            return self.depth_to_bin(complexity_value)
        if self.complexity_metric == "vertex":
            return self.vertex_count_to_bin(complexity_value)
        raise ValueError(f"Unsupported complexity metric: {self.complexity_metric!r}")

    def depth_to_bin(self, ast_depth: int) -> int:
        if ast_depth <= self.depth_bins[0]:
            return self.depth_bins[0]
        for depth_bin in self.depth_bins:
            if ast_depth <= depth_bin:
                return depth_bin
        return self.depth_bins[-1]

    def vertex_count_to_bin(self, node_count: int) -> int:
        if self.vertex_count_thresholds is None:
            raise ValueError("vertex_count_thresholds are only available when complexity_metric='vertex'.")
        for idx, threshold in enumerate(self.vertex_count_thresholds):
            if node_count <= threshold:
                return self.depth_bins[idx]
        return self.depth_bins[-1]

    def complexity_metric_desc(self) -> str:
        if self.complexity_metric == "depth":
            return "AST depth"
        if self.complexity_metric == "vertex":
            return f"AST node count, thresholds={self.vertex_count_thresholds}"
        return self.complexity_metric

    @staticmethod
    def _normalize_complexity_metric(metric: str) -> str:
        metric = str(metric).strip().lower().replace("-", "_")
        aliases = {
            "ast_depth": "depth",
            "depth": "depth",
            "node": "vertex",
            "nodes": "vertex",
            "node_count": "vertex",
            "ast_node_count": "vertex",
            "vertex": "vertex",
            "vertices": "vertex",
        }
        normalized = aliases.get(metric, metric)
        if normalized not in SUPPORTED_COMPLEXITY_METRICS:
            raise ValueError(
                f"Unsupported archive complexity metric {metric!r}. "
                f"Expected one of {SUPPORTED_COMPLEXITY_METRICS}."
            )
        return normalized

    def get(self, descriptor: BehaviorDescriptor | tuple[str, int]) -> EliteRecord | None:
        return self._cells.get(self.get_descriptor(descriptor))

    def update(self, record: EliteRecord) -> bool:
        """
        Insert a record with elitist replacement.

        Returns True if the record occupies the cell after the update, and
        False if it is rejected because the existing elite is better.
        """
        self._validate_record(record)

        incumbent = self._cells.get(record.descriptor)
        record_score, record_corr = self.regularized_quality(record, exclude_descriptor=record.descriptor)
        incumbent_score = None
        incumbent_corr = None
        if incumbent is not None:
            incumbent_score, incumbent_corr = self.regularized_quality(
                incumbent,
                exclude_descriptor=incumbent.descriptor,
            )
        accepted = incumbent is None or record_score > float(incumbent_score)
        self.hist.append(
            EliteArchiveHistory(
                record=record,
                incumbent=incumbent,
                accepted=accepted,
                regularized_quality=record_score,
                corr_penalty=record_corr.penalty,
                corr_reg_coef=record_corr.corr_reg_coef,
                corr_reg_threshold=record_corr.corr_reg_threshold,
                corr_top_k=record_corr.corr_top_k,
                top_k_avg_cross_category_corr=record_corr.top_k_avg_abs_corr,
                max_cross_category_corr=record_corr.max_abs_corr,
                corr_match_factor_name=record_corr.matched_factor_name,
                corr_match_category=record_corr.matched_category,
                corr_match_depth_bin=record_corr.matched_depth_bin,
                corr_compared_count=record_corr.compared_count,
                incumbent_regularized_quality=incumbent_score,
                incumbent_corr_penalty=incumbent_corr.penalty if incumbent_corr is not None else None,
                incumbent_max_cross_category_corr=incumbent_corr.max_abs_corr if incumbent_corr is not None else None,
                incumbent_corr_match_factor_name=incumbent_corr.matched_factor_name
                if incumbent_corr is not None
                else None,
                incumbent_corr_compared_count=incumbent_corr.compared_count if incumbent_corr is not None else None,
            )
        )
        if accepted:
            if incumbent is not None:
                evict_archive_factor_values(incumbent.task)
            self._cells[record.descriptor] = record
            return True
        evict_archive_factor_values(record.task)
        return False

    def sample_parent(self, *, weighted: bool = False) -> EliteRecord:
        """Sample one elite factor as a mutation parent."""
        records = list(self._cells.values())
        if not records:
            raise ValueError("Cannot sample from an empty EliteArchive.")
        if not weighted:
            return self._rng.choice(records)
        return self._weighted_choice(records)

    def sample_pair(self, *, weighted: bool = False) -> tuple[EliteRecord, EliteRecord]:
        """Sample two different elite factors as crossover parents."""
        records = list(self._cells.values())
        if len(records) < 2:
            raise ValueError("Need at least two elites to sample a parent pair.")
        first = self.sample_parent(weighted=weighted)
        rest = [record for record in records if record.descriptor != first.descriptor]
        if not weighted:
            return first, self._rng.choice(rest)
        return first, self._weighted_choice(rest)

    def occupied_descriptors(self) -> list[BehaviorDescriptor]:
        return list(self._cells.keys())

    def records(self) -> list[EliteRecord]:
        return list(self._cells.values())

    def regularized_quality(
        self,
        record: EliteRecord,
        *,
        exclude_descriptor: BehaviorDescriptor | tuple[str, int] | None = None,
    ) -> tuple[float, ArchiveCorrRegularization]:
        corr_regularization = self.cross_category_corr_regularization(
            record,
            exclude_descriptor=exclude_descriptor,
        )
        return float(record.quality) - corr_regularization.penalty, corr_regularization

    def cross_category_corr_regularization(
        self,
        record: EliteRecord,
        *,
        exclude_descriptor: BehaviorDescriptor | tuple[str, int] | None = None,
    ) -> ArchiveCorrRegularization:
        if DEFAULT_ARCHIVE_CORR_REG_COEF <= 0:
            return ArchiveCorrRegularization(reason="corr regularization coefficient is non-positive")

        exclude = self.get_descriptor(exclude_descriptor) if exclude_descriptor is not None else None
        proposed_values = self.get_record_factor_values(record)
        if proposed_values is None:
            return ArchiveCorrRegularization(reason="missing proposed factor values")

        best_corr: float | None = None
        best_record: EliteRecord | None = None
        corr_items: list[tuple[float, EliteRecord]] = []
        compared_count = 0
        for existing in self._cells.values():
            if exclude is not None and existing.descriptor == exclude:
                continue
            if existing.category == record.category:
                continue

            existing_values = self.get_record_factor_values(existing)
            if existing_values is None:
                continue
            corr = self.factor_value_corr(proposed_values, existing_values)
            if corr is None:
                continue
            compared_count += 1
            abs_corr = abs(corr)
            corr_items.append((abs_corr, existing))
            if best_corr is None or abs_corr > best_corr:
                best_corr = abs_corr
                best_record = existing

        if not corr_items:
            return ArchiveCorrRegularization(
                compared_count=compared_count,
                reason="no comparable cross-category factor values",
            )

        threshold = min(1.0, max(0.0, DEFAULT_ARCHIVE_CORR_REG_THRESHOLD))
        sorted_corr_items = sorted(corr_items, key=lambda item: item[0], reverse=True)
        top_k = min(len(sorted_corr_items), DEFAULT_ARCHIVE_CORR_REG_TOP_K)
        top_corrs = [corr for corr, _ in sorted_corr_items[:top_k]]
        top_k_avg_corr = sum(top_corrs) / top_k
        mean_excess_squared = sum(max(0.0, corr - threshold) ** 2 for corr in top_corrs) / top_k
        return ArchiveCorrRegularization(
            penalty=float(DEFAULT_ARCHIVE_CORR_REG_COEF * mean_excess_squared),
            corr_reg_threshold=float(threshold),
            corr_top_k=int(top_k),
            top_k_avg_abs_corr=float(top_k_avg_corr),
            max_abs_corr=float(best_corr),
            matched_factor_name=best_record.factor_name if best_record else None,
            matched_category=best_record.category if best_record else None,
            matched_depth_bin=best_record.depth_bin if best_record else None,
            compared_count=compared_count,
        )

    @staticmethod
    def get_record_factor_values(record: EliteRecord):
        key = getattr(record.task, "archive_factor_values_key", None)
        if key:
            values = _ARCHIVE_FACTOR_VALUE_CACHE.get(key)
            if values is not None:
                return values
        values_path = getattr(record.task, "archive_factor_values_path", None)
        if values_path:
            try:
                import pandas as pd

                path = Path(values_path).expanduser()
                if path.exists():
                    values = pd.read_pickle(path)
                    key = key or str(path)
                    _ARCHIVE_FACTOR_VALUE_CACHE[key] = values
                    setattr(record.task, "archive_factor_values_key", key)
                    return values
            except Exception:
                return None
        return None

    @staticmethod
    def factor_value_corr(values_a, values_b) -> float | None:
        import numpy as np
        import pandas as pd

        if values_a is None or values_b is None:
            return None
        series_a = values_a if isinstance(values_a, pd.Series) else pd.Series(values_a)
        series_b = values_b if isinstance(values_b, pd.Series) else pd.Series(values_b)
        if series_a.empty or series_b.empty:
            return None

        if series_a.index.equals(series_b.index):
            try:
                arr_a = pd.to_numeric(series_a, errors="coerce").to_numpy(dtype=np.float64, copy=False)
                arr_b = pd.to_numeric(series_b, errors="coerce").to_numpy(dtype=np.float64, copy=False)
            except (TypeError, ValueError):
                return None
            mask = np.isfinite(arr_a) & np.isfinite(arr_b)
            if int(mask.sum()) < 2:
                return None
            arr_a = arr_a[mask]
            arr_b = arr_b[mask]
        else:
            pair = pd.concat(
                [
                    pd.to_numeric(series_a, errors="coerce").rename("a"),
                    pd.to_numeric(series_b, errors="coerce").rename("b"),
                ],
                axis=1,
                join="inner",
            ).replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 2:
                return None
            arr_a = pair["a"].to_numpy(dtype=np.float64, copy=False)
            arr_b = pair["b"].to_numpy(dtype=np.float64, copy=False)

        arr_a = arr_a - arr_a.mean()
        arr_b = arr_b - arr_b.mean()
        denom = float(np.sqrt(np.dot(arr_a, arr_a) * np.dot(arr_b, arr_b)))
        if denom <= 0:
            return None
        corr = float(np.dot(arr_a, arr_b) / denom)
        if not isfinite(corr):
            return None
        return corr

    def cross_category_similarity_match(self, record: EliteRecord) -> ArchiveSimilarityMatch | None:
        expression = record.factor_expression
        if not expression:
            return None

        try:
            from alphaagent.components.coder.factor_coder.factor_ast import compare_expressions, count_all_nodes

            proposed_node_count = count_all_nodes(expression)
        except Exception:
            return None

        best_match: ArchiveSimilarityMatch | None = None
        for existing in self._cells.values():
            if existing.category == record.category or not existing.factor_expression:
                continue
            try:
                match = compare_expressions(expression, existing.factor_expression)
            except Exception:
                continue
            if match is None:
                continue

            candidate = ArchiveSimilarityMatch(
                record=existing,
                duplicated_subtree_size=match.size,
                duplicated_subtree=str(match.root1),
                proposed_node_count=proposed_node_count,
            )
            if best_match is None or candidate.duplicated_subtree_size > best_match.duplicated_subtree_size:
                best_match = candidate

        return best_match

    def coverage(self) -> float:
        if self.total_cells == 0:
            return 0.0
        return len(self._cells) / self.total_cells

    def qd_score(self) -> float:
        return sum(record.quality for record in self._cells.values())

    def best(self) -> EliteRecord | None:
        if not self._cells:
            return None
        return max(self._cells.values(), key=lambda record: record.quality)

    def to_records(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._cells.values()]

    def history_records(self) -> list[dict[str, Any]]:
        return [history.to_dict() for history in self.hist]

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.to_records())

    def history_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.history_records())

    @classmethod
    def from_records(
        cls,
        records: Iterable[EliteRecord],
        categories: Sequence[str] = DEFAULT_FACTOR_CATEGORIES,
        depth_bins: Sequence[int] = DEFAULT_DEPTH_BINS,
    ) -> EliteArchive:
        archive = cls(categories=categories, depth_bins=depth_bins)
        for record in records:
            archive.update(record)
        return archive

    def _validate_record(self, record: EliteRecord) -> None:
        if record.descriptor.category not in self.categories:
            raise ValueError(
                f"Record category {record.descriptor.category!r} is not in archive categories {self.categories}."
            )
        if record.descriptor.depth_bin not in self.depth_bins:
            raise ValueError(
                f"Record depth_bin {record.descriptor.depth_bin!r} is not in archive depth bins {self.depth_bins}."
            )
        if not isfinite(float(record.quality)):
            raise ValueError(f"Record quality must be a finite number, got {record.quality!r}.")

    def _weighted_choice(self, records: Sequence[EliteRecord]) -> EliteRecord:
        min_quality = min(record.quality for record in records)
        weights = [(record.quality - min_quality) + 1e-12 for record in records]
        total = sum(weights)
        if total <= 0:
            return self._rng.choice(list(records))
        threshold = self._rng.random() * total
        running = 0.0
        for record, weight in zip(records, weights):
            running += weight
            if running >= threshold:
                return record
        return records[-1]


def update_archive_from_experiment(archive: EliteArchive, exp, log=None) -> None:
    for task in exp.sub_tasks:
        descriptor = get_task_descriptor(archive, task)
        if descriptor is None:
            if log is not None:
                log.warning(f"Skip archive update for {task.factor_name}: missing factor category or complexity descriptor.")
            continue

        quality = get_task_quality(exp, task)
        if quality is None:
            if log is not None:
                log.warning(f"Skip archive update for {task.factor_name}: missing quality metric.")
            continue

        attach_task_quality_metadata_from_exp(exp, task)
        record = EliteRecord.from_task(task, descriptor=descriptor, quality=quality)
        accepted = archive.update(record)
        if log is not None:
            latest_history = archive.hist[-1] if archive.hist else None
            rejection_reason = (
                getattr(latest_history, "rejection_reason", None)
                if latest_history is not None and latest_history.record is record
                else None
            )
            rejection_text = "" if not rejection_reason else f", rejection_reason={rejection_reason}"
            corr_text = ""
            if latest_history is not None and latest_history.record is record:
                corr_text = (
                    f", regularized_quality={getattr(latest_history, 'regularized_quality', None)}, "
                    f"corr_penalty={getattr(latest_history, 'corr_penalty', None)}, "
                    f"corr_reg_threshold={getattr(latest_history, 'corr_reg_threshold', None)}, "
                    f"corr_top_k={getattr(latest_history, 'corr_top_k', None)}, "
                    f"top_k_avg_cross_category_corr="
                    f"{getattr(latest_history, 'top_k_avg_cross_category_corr', None)}, "
                    f"max_cross_category_corr={getattr(latest_history, 'max_cross_category_corr', None)}, "
                    f"corr_match={getattr(latest_history, 'corr_match_factor_name', None)}, "
                    f"incumbent_regularized_quality="
                    f"{getattr(latest_history, 'incumbent_regularized_quality', None)}, "
                    f"incumbent_corr_penalty={getattr(latest_history, 'incumbent_corr_penalty', None)}"
                )
            log.info(
                f"Elite archive update for {task.factor_name}: "
                f"cell=({descriptor.category}, {descriptor.depth_bin}), "
                f"metric={archive.complexity_metric}, quality_metric={DEFAULT_QUALITY_METRIC}, "
                f"quality_segment={getattr(record, 'quality_segment', None)}, "
                f"quality={quality}, validation_Rank IC={getattr(record, 'validation_rank_ic', None)}, "
                f"accepted={accepted}{corr_text}{rejection_text}"
            )


def rebuild_archive_from_trace_history(trace, log=None) -> None:
    """Rebuild a loaded EliteAlpha archive under the current bin/quality rules."""
    current_archive = getattr(trace, "archive", None)
    if current_archive is None:
        return

    archive = EliteArchive(
        categories=getattr(current_archive, "categories", DEFAULT_FACTOR_CATEGORIES),
        depth_bins=DEFAULT_DEPTH_BINS,
        complexity_metric=getattr(current_archive, "complexity_metric", DEFAULT_COMPLEXITY_METRIC),
        vertex_count_thresholds=getattr(current_archive, "vertex_count_thresholds", None),
    )
    attempts = 0
    accepted = 0
    skipped = 0
    for _, exp, _ in getattr(trace, "hist", []) or []:
        for task in getattr(exp, "sub_tasks", []) or []:
            descriptor = get_task_descriptor(archive, task)
            quality = get_task_quality(exp, task)
            if descriptor is None or quality is None:
                skipped += 1
                continue
            attach_task_quality_metadata_from_exp(exp, task)
            attempts += 1
            accepted += int(archive.update(EliteRecord.from_task(task, descriptor=descriptor, quality=quality)))

    trace.archive = archive
    if log is not None:
        log.info(
            "Rebuilt EliteAlpha archive from trace history: "
            f"attempts={attempts}, accepted={accepted}, skipped={skipped}, "
            f"bins={archive.depth_bins}, quality_metric={DEFAULT_QUALITY_METRIC}"
        )


def get_task_descriptor(archive: EliteArchive, task) -> BehaviorDescriptor | None:
    descriptor = getattr(task, "elite_descriptor", None)
    category = (
        getattr(task, "factor_category", None)
        or getattr(task, "elite_category", None)
        or getattr(task, "category", None)
    )
    if category is None and isinstance(descriptor, BehaviorDescriptor):
        category = descriptor.category
    if category is None and isinstance(descriptor, tuple) and len(descriptor) == 2:
        category = descriptor[0]
    if category is None:
        return None

    complexity_value = get_task_complexity_value(archive, task)
    if complexity_value is not None:
        if archive.complexity_metric == "depth" and int(complexity_value) > int(archive.depth_bins[-1]):
            return None
        return archive.make_descriptor(str(category), int(complexity_value))

    depth_bin = (
        getattr(task, "depth_bin", None)
        or getattr(task, "elite_depth_bin", None)
        or getattr(task, "elite_complexity_bin", None)
    )
    if depth_bin is not None:
        depth_bin = int(depth_bin)
        if depth_bin not in archive.depth_bins:
            return None
        return BehaviorDescriptor(category=archive.normalize_category(str(category)), depth_bin=depth_bin)

    if isinstance(descriptor, BehaviorDescriptor):
        if descriptor.depth_bin not in archive.depth_bins:
            return None
        return BehaviorDescriptor(
            category=archive.normalize_category(str(descriptor.category)),
            depth_bin=int(descriptor.depth_bin),
        )
    if isinstance(descriptor, tuple) and len(descriptor) == 2:
        descriptor = archive.get_descriptor(descriptor)
        if descriptor.depth_bin not in archive.depth_bins:
            return None
        return descriptor

    return None


def get_task_complexity_value(archive: EliteArchive, task) -> int | None:
    if archive.complexity_metric == "vertex":
        value = (
            getattr(task, "factor_ast_node_count", None)
            or getattr(task, "factor_complexity_value", None)
            or getattr(task, "ast_node_count", None)
            or getattr(task, "node_count", None)
            or getattr(task, "vertex_count", None)
        )
    else:
        value = (
            getattr(task, "ast_depth", None)
            or getattr(task, "factor_ast_depth", None)
            or getattr(task, "factor_complexity_value", None)
        )
    if value is None:
        return None
    return int(value)


def get_task_quality(exp, task) -> float | None:
    sub_quality = get_sub_result_quality(exp, task.factor_name)
    if sub_quality is not None:
        return sub_quality
    return None


def attach_task_quality_metadata_from_exp(exp, task) -> None:
    sub_result = getattr(exp, "sub_results", {}).get(task.factor_name)
    if not isinstance(sub_result, Mapping):
        return
    task.archive_quality_segment = sub_result.get("quality_segment", "train")
    task.train_rank_ic = sub_result.get("train_Rank IC", sub_result.get("Rank IC"))
    task.train_rank_icir = sub_result.get("train_Rank ICIR", sub_result.get("Rank ICIR"))
    task.validation_rank_ic = sub_result.get("validation_Rank IC")
    task.validation_rank_icir = sub_result.get("validation_Rank ICIR")
    task.validation_raw_rank_ic = sub_result.get("validation_raw_Rank IC")


def normalize_quality(value: Any) -> float | None:
    try:
        quality = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(quality):
        return None
    return quality


def get_sub_result_quality(exp, factor_name: str) -> float | None:
    sub_result = getattr(exp, "sub_results", {}).get(factor_name)
    if sub_result is None:
        return None
    if isinstance(sub_result, (int, float)):
        return normalize_quality(sub_result)
    if isinstance(sub_result, Mapping):
        return quality_from_mapping(sub_result)
    return None


def get_result_quality(result) -> float | None:
    if result is None:
        return None
    if isinstance(result, Mapping):
        return quality_from_mapping(result)

    for key in _QUALITY_KEY_PRIORITY:
        try:
            if key in result.index:
                return normalize_quality(result.loc[key])
        except AttributeError:
            break
    return None


def quality_from_mapping(values: Mapping) -> float | None:
    for key in _QUALITY_KEY_PRIORITY:
        if key in values:
            return normalize_quality(values[key])
    return None


def format_archive_view(archive: EliteArchive) -> str:
    records = archive.to_records()
    detail_records = _sort_records_by_validation_priority(records)
    detail_numbers = _build_detail_numbers(detail_records)
    lines = [
        "EliteAlpha Archive",
        f"Complexity metric: {archive.complexity_metric_desc()}",
        f"Bins: {archive.depth_bins}",
        f"Quality metric: train {DEFAULT_QUALITY_METRIC}",
        f"Coverage: {len(archive)}/{archive.total_cells} = {archive.coverage():.2%}",
        f"QD score: {archive.qd_score()}",
        "",
        format_archive_matrix(
            records,
            categories=archive.categories,
            depth_bins=archive.depth_bins,
            detail_numbers=detail_numbers,
        ),
        "",
        format_validation_priority(records),
        "",
        format_archive_details(detail_records, detail_numbers=detail_numbers),
    ]
    return "\n".join(lines)


def format_archive_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    categories: Sequence[str] = DEFAULT_FACTOR_CATEGORIES,
    depth_bins: Sequence[int] = DEFAULT_DEPTH_BINS,
    detail_numbers: Mapping[tuple[str, int], int] | None = None,
) -> str:
    cells = {}
    detail_numbers = dict(detail_numbers or {})
    for idx, record in enumerate(records, start=1):
        key = (record.get("category"), int(record.get("depth_bin")))
        cells[key] = record
        detail_numbers.setdefault(key, idx)

    label_width = max(len(category) for category in categories)
    cell_width = 18
    header = " " * (label_width + 2) + "".join(f"bin={depth}".center(cell_width) for depth in depth_bins)
    lines = ["Archive Matrix", header, "-" * len(header)]
    for category in categories:
        top_row = [f"{category:<{label_width}}  "]
        bottom_row = [" " * (label_width + 2)]
        has_record = False
        for depth in depth_bins:
            record = cells.get((category, int(depth)))
            if record is None:
                top_cell = "."
                bottom_cell = ""
            else:
                has_record = True
                number = detail_numbers[(category, int(depth))]
                top_cell = f"[{number}]"
                bottom_cell = f"quality={_format_quality(record.get('quality'))}"
            top_row.append(top_cell.center(cell_width))
            bottom_row.append(bottom_cell.center(cell_width))
        lines.append("".join(top_row))
        if has_record:
            lines.append("".join(bottom_row))
    return "\n".join(lines)


def format_archive_details(
    records: Sequence[Mapping[str, Any]],
    *,
    detail_numbers: Mapping[tuple[str, int], int] | None = None,
) -> str:
    if not records:
        return "Details\n(empty)"

    lines = ["Details"]
    for idx, record in enumerate(records, start=1):
        key = (record.get("category"), int(record.get("depth_bin")))
        number = detail_numbers.get(key, idx) if detail_numbers else idx
        metric = record.get("factor_complexity_metric")
        metric_value = record.get("factor_complexity_value")
        optional_stats = []
        if record.get("factor_ast_depth") is not None:
            optional_stats.append(f"ast_depth={record.get('factor_ast_depth')}")
        if record.get("factor_ast_node_count") is not None:
            optional_stats.append(f"ast_nodes={record.get('factor_ast_node_count')}")
        optional_text = "" if not optional_stats else " | " + " | ".join(optional_stats)
        quality_metric = record.get("quality_metric") or DEFAULT_QUALITY_METRIC
        validation_rank_ic = record.get("validation_Rank IC")
        validation_text = ""
        if validation_rank_ic is not None:
            validation_text = f" | validation_Rank IC={_format_quality(validation_rank_ic)}"
        lines.append(
            f"[{number}] {record.get('factor_name')} "
            f"| cell=({record.get('category')}, {record.get('depth_bin')}) "
            f"| metric={metric} "
            f"| metric_value={metric_value}"
            f"{optional_text} "
            f"| quality_metric=train {quality_metric} "
            f"| quality={record.get('quality')}"
            f"{validation_text}"
        )
        expression = record.get("factor_expression")
        if expression:
            lines.append(f"    expr: {expression}")
        description = record.get("factor_description")
        if description:
            lines.append(f"    desc: {_shorten(description, 180)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_detail_numbers(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], int]:
    return {
        (record.get("category"), int(record.get("depth_bin"))): idx
        for idx, record in enumerate(records, start=1)
    }


def _sort_records_by_validation_priority(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            _optional_float(record.get("validation_Rank IC")) is not None,
            _optional_float(record.get("validation_Rank IC"))
            if _optional_float(record.get("validation_Rank IC")) is not None
            else float("-inf"),
            _optional_float(record.get("quality")) if _optional_float(record.get("quality")) is not None else float("-inf"),
        ),
        reverse=True,
    )


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_validation_priority(records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        return "Validation Priority\n(empty)"

    ranked = []
    for record in records:
        validation_rank_ic = record.get("validation_Rank IC")
        try:
            validation_rank_ic = float(validation_rank_ic)
        except (TypeError, ValueError):
            validation_rank_ic = None
        ranked.append((validation_rank_ic, record))

    ranked.sort(
        key=lambda item: (
            item[0] is not None,
            item[0] if item[0] is not None else float("-inf"),
        ),
        reverse=True,
    )

    lines = ["Validation Priority (no corr/depth regularization)"]
    for idx, (validation_rank_ic, record) in enumerate(ranked, start=1):
        lines.append(
            f"[{idx}] {record.get('factor_name')} "
            f"| validation_Rank IC={_format_quality(validation_rank_ic)} "
            f"| train_Rank IC={_format_quality(record.get('quality'))} "
            f"| cell=({record.get('category')}, {record.get('depth_bin')})"
        )
    return "\n".join(lines)


def _shorten(text: Any, max_len: int) -> str:
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_quality(value: Any, digits: int = 5) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)
