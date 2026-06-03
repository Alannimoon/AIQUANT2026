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
            "quality": self.quality,
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
            "quality": self.record.quality,
            "accepted": self.accepted,
            "incumbent_factor_name": self.incumbent.factor_name if self.incumbent else None,
            "incumbent_quality": self.incumbent.quality if self.incumbent else None,
        }


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
        accepted = incumbent is None or record.quality > incumbent.quality
        self.hist.append(EliteArchiveHistory(record=record, incumbent=incumbent, accepted=accepted))
        if accepted:
            self._cells[record.descriptor] = record
            return True
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

        accepted = archive.update(EliteRecord.from_task(task, descriptor=descriptor, quality=quality))
        if log is not None:
            log.info(
                f"Elite archive update for {task.factor_name}: "
                f"cell=({descriptor.category}, {descriptor.depth_bin}), "
                f"metric={archive.complexity_metric}, quality={quality}, accepted={accepted}"
            )


def get_task_descriptor(archive: EliteArchive, task) -> BehaviorDescriptor | None:
    descriptor = getattr(task, "elite_descriptor", None)
    if isinstance(descriptor, BehaviorDescriptor):
        return descriptor
    if isinstance(descriptor, tuple) and len(descriptor) == 2:
        return archive.get_descriptor(descriptor)

    category = (
        getattr(task, "factor_category", None)
        or getattr(task, "elite_category", None)
        or getattr(task, "category", None)
    )
    if category is None:
        return None

    depth_bin = (
        getattr(task, "depth_bin", None)
        or getattr(task, "elite_depth_bin", None)
        or getattr(task, "elite_complexity_bin", None)
    )
    if depth_bin is not None:
        return BehaviorDescriptor(category=archive.normalize_category(str(category)), depth_bin=int(depth_bin))

    complexity_value = get_task_complexity_value(archive, task)
    if complexity_value is None:
        return None
    return archive.make_descriptor(str(category), int(complexity_value))


def get_task_complexity_value(archive: EliteArchive, task) -> int | None:
    if archive.complexity_metric == "vertex":
        value = (
            getattr(task, "factor_ast_node_count", None)
            or getattr(task, "ast_node_count", None)
            or getattr(task, "node_count", None)
            or getattr(task, "vertex_count", None)
        )
    else:
        value = getattr(task, "ast_depth", None) or getattr(task, "factor_ast_depth", None)
    if value is None:
        return None
    return int(value)


def get_task_quality(exp, task) -> float | None:
    sub_quality = get_sub_result_quality(exp, task.factor_name)
    if sub_quality is not None:
        return sub_quality
    if len(getattr(exp, "sub_tasks", []) or []) > 1:
        return None
    return get_result_quality(exp.result)


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

    for key in ("IC", "Rank IC", "RankIC", "ic", "rank_ic"):
        try:
            if key in result.index:
                return normalize_quality(result.loc[key])
        except AttributeError:
            break
    return None


def quality_from_mapping(values: Mapping) -> float | None:
    for key in ("IC", "Rank IC", "RankIC", "ic", "rank_ic"):
        if key in values:
            return normalize_quality(values[key])
    return None


def format_archive_view(archive: EliteArchive) -> str:
    records = archive.to_records()
    lines = [
        "EliteAlpha Archive",
        f"Complexity metric: {archive.complexity_metric_desc()}",
        f"Bins: {archive.depth_bins}",
        f"Coverage: {len(archive)}/{archive.total_cells} = {archive.coverage():.2%}",
        f"QD score: {archive.qd_score()}",
        "",
        format_archive_matrix(records, categories=archive.categories, depth_bins=archive.depth_bins),
        "",
        format_archive_details(records),
    ]
    return "\n".join(lines)


def format_archive_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    categories: Sequence[str] = DEFAULT_FACTOR_CATEGORIES,
    depth_bins: Sequence[int] = DEFAULT_DEPTH_BINS,
) -> str:
    cells = {}
    detail_numbers = {}
    for idx, record in enumerate(records, start=1):
        key = (record.get("category"), int(record.get("depth_bin")))
        cells[key] = record
        detail_numbers[key] = idx

    label_width = max(len(category) for category in categories)
    cell_width = 12
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
                bottom_cell = f"q={_format_quality(record.get('quality'))}"
            top_row.append(top_cell.center(cell_width))
            bottom_row.append(bottom_cell.center(cell_width))
        lines.append("".join(top_row))
        if has_record:
            lines.append("".join(bottom_row))
    return "\n".join(lines)


def format_archive_details(records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        return "Details\n(empty)"

    lines = ["Details"]
    for idx, record in enumerate(records, start=1):
        metric = record.get("factor_complexity_metric")
        metric_value = record.get("factor_complexity_value")
        optional_stats = []
        if record.get("factor_ast_depth") is not None:
            optional_stats.append(f"ast_depth={record.get('factor_ast_depth')}")
        if record.get("factor_ast_node_count") is not None:
            optional_stats.append(f"ast_nodes={record.get('factor_ast_node_count')}")
        optional_text = "" if not optional_stats else " | " + " | ".join(optional_stats)
        lines.append(
            f"[{idx}] {record.get('factor_name')} "
            f"| cell=({record.get('category')}, {record.get('depth_bin')}) "
            f"| metric={metric} "
            f"| metric_value={metric_value}"
            f"{optional_text} "
            f"| quality={record.get('quality')}"
        )
        expression = record.get("factor_expression")
        if expression:
            lines.append(f"    expr: {expression}")
        description = record.get("factor_description")
        if description:
            lines.append(f"    desc: {_shorten(description, 180)}")
        lines.append("")
    return "\n".join(lines).rstrip()


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
