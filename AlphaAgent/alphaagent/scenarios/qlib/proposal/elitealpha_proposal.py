import json
import random
import re
from pathlib import Path
from typing import Any, List, Tuple

from jinja2 import Environment, StrictUndefined

from alphaagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
from alphaagent.components.coder.factor_coder.factor_ast import (
    BinaryOpNode,
    ConditionalNode,
    FunctionNode,
    Node,
    parse_expression,
)
from alphaagent.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from alphaagent.core.experiment import Experiment
from alphaagent.core.prompts import Prompts
from alphaagent.core.proposal import Hypothesis, Trace
from alphaagent.core.scenario import Scenario
from alphaagent.log import logger
from alphaagent.oai.llm_utils import APIBackend
from alphaagent.scenarios.qlib.archive import DEFAULT_QUALITY_METRIC, EliteArchive
from alphaagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from alphaagent.scenarios.qlib.proposal.factor_proposal import AlphaAgentHypothesis
from alphaagent.scenarios.qlib.regulator.factor_regulator import FactorRegulator


alphaagent_prompt_dict = Prompts(file_path=Path(__file__).parent.parent / "prompts_alphaagent.yaml")
ELITE_ALPHA_MAX_AST_DEPTH = 5
ELITE_ALPHA_CREATIVE_INNOVATION_PROBABILITY = 0.15
ELITE_ALPHA_CANDIDATE_COUNT = 5
ELITE_ALPHA_CREATIVE_FEEDBACK_HISTORY_LIMIT = 20
ELITE_ALPHA_FEEDBACK_FIELD_MAX_CHARS = 700


def _single_line_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    return " ".join(text.split())


def _select_hypothesis_response(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "hypothesis" in payload:
        return payload

    candidates = None
    if isinstance(payload, dict):
        candidates = payload.get("candidates")
    elif isinstance(payload, list):
        candidates = payload

    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("hypothesis"):
                logger.warning(
                    "EliteAlpha hypothesis response contained a candidates list; "
                    "using the first candidate with `hypothesis`."
                )
                return candidate

    return payload


class EliteAlphaHypothesis(AlphaAgentHypothesis):
    def __init__(
        self,
        hypothesis: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        concise_specification: str,
        elite_search_plan: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            hypothesis=hypothesis,
            concise_observation=concise_observation,
            concise_justification=concise_justification,
            concise_knowledge=concise_knowledge,
            concise_specification=concise_specification,
        )
        self.elite_search_plan = elite_search_plan or {}

    def __str__(self) -> str:
        base = super().__str__()
        return f"""{base}
                EliteAlpha Search Plan: {_sanitize_search_plan_for_prompt(self.elite_search_plan)}
                """


class EliteAlphaHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario, potential_direction: str = None) -> Tuple[dict, bool]:
        super().__init__(scen)
        self.potential_direction = potential_direction
        self._last_search_plan: dict[str, Any] = {}

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        archive = _require_archive(trace)
        search_plan = _build_elite_search_plan(archive, trace, self.potential_direction)
        self._last_search_plan = search_plan

        context_parts = [
            _format_archive_context(archive),
            _format_search_plan(search_plan),
            _format_history_for_search_plan(trace, search_plan),
        ]

        if len(trace.hist) == 0 and self.potential_direction is not None:
            direction_context = (
                Environment(undefined=StrictUndefined)
                .from_string(alphaagent_prompt_dict["potential_direction_transformation"])
                .render(potential_direction=self.potential_direction)
            )
            context_parts.append(direction_context)

        hypothesis_and_feedback = "\n\n".join(part for part in context_parts if part)

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": None,
            "hypothesis_output_format": alphaagent_prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": _elite_hypothesis_specification(
                alphaagent_prompt_dict["factor_hypothesis_specification"]
            ),
        }
        return context_dict, True

    def convert_response(self, response: str) -> EliteAlphaHypothesis:
        response_dict = _select_hypothesis_response(json.loads(response))
        if not isinstance(response_dict, dict):
            raise ValueError(f"EliteAlpha hypothesis response must be a JSON object, got {type(response_dict).__name__}.")

        hypothesis = response_dict.get("hypothesis")
        if not hypothesis:
            raise ValueError(
                "EliteAlpha hypothesis response is missing `hypothesis`; "
                f"available keys={list(response_dict.keys())}."
            )

        return EliteAlphaHypothesis(
            hypothesis=_single_line_text(hypothesis),
            concise_observation=_single_line_text(
                response_dict.get("concise_observation") or response_dict.get("observation") or ""
            ),
            concise_knowledge=_single_line_text(
                response_dict.get("concise_knowledge") or response_dict.get("knowledge") or ""
            ),
            concise_justification=_single_line_text(
                response_dict.get("concise_justification") or response_dict.get("justification") or ""
            ),
            concise_specification=_single_line_text(
                response_dict.get("concise_specification") or response_dict.get("specification") or ""
            ),
            elite_search_plan=self._last_search_plan,
        )

    def gen(self, trace: Trace) -> EliteAlphaHypothesis:
        context_dict, json_flag = self.prepare_context(trace)
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(alphaagent_prompt_dict["hypothesis_gen"]["system_prompt"])
            .render(
                targets=self.targets,
                scenario=self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                hypothesis_output_format=context_dict["hypothesis_output_format"],
                hypothesis_specification=context_dict["hypothesis_specification"],
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(alphaagent_prompt_dict["hypothesis_gen"]["user_prompt"])
            .render(
                targets=self.targets,
                hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                RAG=context_dict["RAG"],
                round=len(trace.hist),
            )
        )

        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)
        return self.convert_response(resp)


class EliteAlphaHypothesis2FactorExpression(FactorHypothesis2Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor_regulator = FactorRegulator(depth_cap=ELITE_ALPHA_MAX_AST_DEPTH)
        self.max_regeneration_attempts = 10

    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]:
        archive = _require_archive(trace)
        search_plan = getattr(hypothesis, "elite_search_plan", None) or _build_elite_search_plan(archive, trace, None)

        scenario = trace.scen.get_scenario_all_desc()
        experiment_output_format = _elite_experiment_output_format(
            alphaagent_prompt_dict["factor_experiment_output_format"],
            archive,
        )
        function_lib_description = alphaagent_prompt_dict["function_lib_description"]

        hypothesis_and_feedback = "\n\n".join(
            [
                _format_archive_context(archive),
                _format_search_plan(search_plan),
                _format_history_for_search_plan(trace, search_plan),
                _format_archive_targets(archive),
            ]
        )

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "function_lib_description": function_lib_description,
            "experiment_output_format": experiment_output_format,
            "target_list": [],
            "RAG": None,
            "search_plan": search_plan,
        }, True

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        archive = _require_archive(trace)
        _sync_regulator_factor_categories(self.factor_regulator, trace)
        context, json_flag = self.prepare_context(hypothesis, trace)
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(alphaagent_prompt_dict["hypothesis2experiment"]["system_prompt"])
            .render(
                targets=self.targets,
                scenario=trace.scen.background,
                experiment_output_format=context["experiment_output_format"],
            )
        )
        user_prompt = self._render_user_prompt(context)

        response = None
        accepted_response_dict = {}
        expression_rejection_prompt = None

        for attempt_idx in range(self.max_regeneration_attempts):
            response = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)
            response_dict = json.loads(response)
            attempt_accepted = {}

            for factor_name, factor_info in response_dict.items():
                expr = factor_info["expression"]
                expr_for_evaluation = _expression_without_leading_sign(expr)
                factor_category = _resolve_category(
                    archive,
                    factor_info.get("category"),
                    context["search_plan"].get("target_category"),
                    factor_name,
                    factor_info.get("description", ""),
                    factor_info.get("formulation", ""),
                    expr,
                )

                if not self.factor_regulator.is_parsable(expr_for_evaluation):
                    logger.info(f"Skip unparsable EliteAlpha expr from {factor_name}: {expr}")
                    continue

                success, eval_dict = self.factor_regulator.evaluate(
                    expr_for_evaluation,
                    factor_category=factor_category,
                )
                if not success:
                    logger.info(f"Skip unevaluable EliteAlpha expr from {factor_name}: {expr}")
                    continue

                if not self.factor_regulator.is_expression_acceptable(eval_dict):
                    logger.info(
                        "Skip unacceptable EliteAlpha expr from "
                        f"{factor_name}: {expr}; eval={eval_dict}"
                    )
                    expression_rejection_prompt = _append_expression_rejection_feedback(
                        expression_rejection_prompt,
                        expr,
                        eval_dict,
                        duplication_threshold=self.factor_regulator.duplication_threshold,
                        depth_cap=self.factor_regulator.depth_cap,
                    )
                    context["expression_rejection"] = expression_rejection_prompt
                    continue

                attempt_accepted[factor_name] = factor_info

            if attempt_accepted:
                accepted_response_dict.update(attempt_accepted)
                break

            logger.info(
                f"No acceptable EliteAlpha factor expressions in attempt "
                f"{attempt_idx + 1}/{self.max_regeneration_attempts}; retrying..."
            )
            if expression_rejection_prompt is not None:
                user_prompt = self._render_user_prompt(context)

        if not accepted_response_dict:
            raise ValueError("Failed to generate acceptable EliteAlpha factor expressions.")

        proposed_names = list(accepted_response_dict)
        proposed_exprs = [factor_info["expression"] for factor_info in accepted_response_dict.values()]
        proposed_categories = [
            _resolve_category(
                archive,
                factor_info.get("category"),
                context["search_plan"].get("target_category"),
                factor_name,
                factor_info.get("description", ""),
                factor_info.get("formulation", ""),
                factor_info["expression"],
            )
            for factor_name, factor_info in accepted_response_dict.items()
        ]
        self.factor_regulator.add_factor(proposed_names, proposed_exprs, proposed_categories)
        return self.convert_response(json.dumps(accepted_response_dict), trace, context["search_plan"])

    def _render_user_prompt(self, context: dict[str, Any]) -> str:
        return (
            Environment(undefined=StrictUndefined)
            .from_string(alphaagent_prompt_dict["hypothesis2experiment"]["user_prompt"])
            .render(
                targets=self.targets,
                target_hypothesis=context["target_hypothesis"],
                hypothesis_and_feedback=context["hypothesis_and_feedback"],
                function_lib_description=context["function_lib_description"],
                target_list=context["target_list"],
                RAG=context["RAG"],
                expression_rejection=context.get("expression_rejection"),
            )
        )

    def convert_response(
        self,
        response: str,
        trace: Trace,
        search_plan: dict[str, Any] | None = None,
    ) -> FactorExperiment:
        archive = _require_archive(trace)
        response_dict = json.loads(response)
        search_plan = search_plan or {}
        tasks = []

        for factor_name in response_dict:
            factor_info = response_dict[factor_name]
            description = factor_info["description"]
            formulation = factor_info["formulation"]
            expression = factor_info["expression"]
            variables = factor_info.get("variables", {})

            category = _resolve_category(
                archive,
                factor_info.get("category"),
                search_plan.get("target_category"),
                factor_name,
                description,
                formulation,
                expression,
            )
            complexity_value = _resolve_expression_complexity(
                archive,
                factor_info.get("ast_depth"),
                expression,
                search_plan.get("target_depth_bin"),
            )
            descriptor = archive.make_descriptor(category, complexity_value)

            task = FactorTask(
                factor_name=factor_name,
                factor_description=description,
                factor_formulation=formulation,
                factor_expression=expression,
                variables=variables,
            )
            task.factor_category = descriptor.category
            task.factor_complexity_metric = archive.complexity_metric
            task.factor_complexity_value = complexity_value
            if archive.complexity_metric == "depth":
                task.factor_ast_depth = complexity_value
            elif archive.complexity_metric == "vertex":
                task.factor_ast_node_count = complexity_value
            task.elite_depth_bin = descriptor.depth_bin
            task.elite_complexity_bin = descriptor.depth_bin
            task.elite_descriptor = descriptor
            task.elite_generation_mode = search_plan.get("mode")
            task.elite_parent_factors = search_plan.get("parents", [])
            tasks.append(task)

        based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]
        unique_tasks = _filter_duplicate_tasks(tasks, based_experiments)

        exp = QlibFactorExperiment(unique_tasks)
        exp.based_experiments = based_experiments
        exp.is_elitealpha = True
        exp.skip_sota_factor_merge = True
        return exp


def _require_archive(trace: Trace) -> EliteArchive:
    archive = getattr(trace, "archive", None)
    if archive is None:
        raise TypeError("EliteAlpha proposal requires trace.archive. Use EliteAlphaTrace with EliteAlphaLoop.")
    return archive


def _build_elite_search_plan(
    archive: EliteArchive,
    trace: Trace,
    potential_direction: str | None,
) -> dict[str, Any]:
    empty_cells = _archive_empty_cells(archive)
    mode_probabilities = _elite_mode_probabilities(
        archive,
        empty_cells,
        has_feedback=len(trace.hist) > 0,
    )
    mode = _sample_from_distribution(mode_probabilities)

    if mode == "creative_innovation":
        return _make_elite_search_plan(
            archive=archive,
            mode=mode,
            target_category=None,
            target_depth_bin=None,
            parents=[],
            instruction=(
                "Generate fully new factors from sanitized natural-language feedback only. "
                "Do not use, request, infer, or reconstruct any existing factor expression. "
                f"Return exactly {ELITE_ALPHA_CANDIDATE_COUNT} diverse candidates spread across any categories "
                "and AST depths up to the hard cap."
            ),
            potential_direction=potential_direction,
            mode_probabilities=mode_probabilities,
            empty_cell_count=len(empty_cells),
            target_candidate_count=ELITE_ALPHA_CANDIDATE_COUNT,
        )

    if mode == "initialize_empty_cell":
        category, depth_bin = _sample_empty_cell(archive, trace, empty_cells)
        return _make_elite_search_plan(
            archive=archive,
            mode=mode,
            target_category=category,
            target_depth_bin=depth_bin,
            parents=[],
            instruction=(
                "Initialize the requested MAP-Elites cell with a diverse, testable factor. "
                "Use only the archive occupancy and quality signals provided in the prompt."
            ),
            potential_direction=potential_direction,
            mode_probabilities=mode_probabilities,
            empty_cell_count=len(empty_cells),
            target_candidate_count=ELITE_ALPHA_CANDIDATE_COUNT,
        )

    if mode == "crossover":
        left, right = archive.sample_pair(weighted=True)
        target_category = left.category if random.random() < 0.5 else right.category
        target_depth_bin = random.choice((left.depth_bin, right.depth_bin))
        return _make_elite_search_plan(
            archive=archive,
            mode=mode,
            target_category=target_category,
            target_depth_bin=target_depth_bin,
            parents=[
                _sanitize_elite_record_for_prompt(left, include_expression=True),
                _sanitize_elite_record_for_prompt(right, include_expression=True),
            ],
            instruction=(
                "Generate a candidate for the requested cell using only the provided parent expressions, "
                "parent cells, and quality scores. Do not use or infer any other archive expressions."
            ),
            potential_direction=potential_direction,
            mode_probabilities=mode_probabilities,
            empty_cell_count=len(empty_cells),
            target_candidate_count=ELITE_ALPHA_CANDIDATE_COUNT,
        )

    parent = archive.sample_parent(weighted=True)
    return _make_elite_search_plan(
        archive=archive,
        mode="mutation",
        target_category=parent.category,
        target_depth_bin=parent.depth_bin,
        parents=[_sanitize_elite_record_for_prompt(parent, include_expression=True)],
        instruction=(
            "Generate a mutation candidate using only the provided parent expression, parent cell, "
            "and quality score. Do not use or infer any other archive expressions."
        ),
        potential_direction=potential_direction,
        mode_probabilities=mode_probabilities,
        empty_cell_count=len(empty_cells),
        target_candidate_count=ELITE_ALPHA_CANDIDATE_COUNT,
    )


def _make_elite_search_plan(
    *,
    archive: EliteArchive,
    mode: str,
    target_category: str | None,
    target_depth_bin: int | None,
    parents: list[dict[str, Any]],
    instruction: str,
    potential_direction: str | None,
    mode_probabilities: dict[str, float],
    empty_cell_count: int,
    target_candidate_count: int | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "mode_probabilities": _round_probabilities(mode_probabilities),
        "archive_coverage": archive.coverage(),
        "empty_cell_count": empty_cell_count,
        "target_category": target_category,
        "target_depth_bin": target_depth_bin,
        "target_complexity_bin": target_depth_bin,
        "target_complexity_metric": archive.complexity_metric,
        "parents": parents,
        "instruction": instruction,
        "potential_direction": potential_direction,
        "target_candidate_count": target_candidate_count,
    }


def _archive_empty_cells(archive: EliteArchive) -> list[tuple[str, int]]:
    occupied = {(descriptor.category, int(descriptor.depth_bin)) for descriptor in archive.occupied_descriptors()}
    return [
        (category, int(depth_bin))
        for category in archive.categories
        for depth_bin in archive.depth_bins
        if (category, int(depth_bin)) not in occupied
    ]


def _elite_mode_probabilities(
    archive: EliteArchive,
    empty_cells: list[tuple[str, int]],
    *,
    has_feedback: bool = False,
) -> dict[str, float]:
    if len(archive) == 0:
        return _normalize_probabilities(
            {
                "initialize_empty_cell": 0.75 if empty_cells else 0.0,
                "creative_innovation": 0.25 if has_feedback else 0.0,
                "mutation": 0.0,
                "crossover": 0.0,
            },
            fallback_mode="initialize_empty_cell",
        )

    coverage = archive.coverage()
    creative_innovation = ELITE_ALPHA_CREATIVE_INNOVATION_PROBABILITY if has_feedback else 0.0
    probabilities = {
        "initialize_empty_cell": 0.60 * (1.0 - coverage) if empty_cells else 0.0,
        "creative_innovation": creative_innovation,
        "mutation": 0.0,
        "crossover": 0.10 + 0.25 * coverage if len(archive) >= 2 else 0.0,
    }
    probabilities["mutation"] = max(
        0.0,
        1.0
        - probabilities["initialize_empty_cell"]
        - probabilities["creative_innovation"]
        - probabilities["crossover"],
    )
    return _normalize_probabilities(probabilities, fallback_mode="mutation")


def _normalize_probabilities(probabilities: dict[str, float], fallback_mode: str) -> dict[str, float]:
    clipped = {mode: max(0.0, float(probability)) for mode, probability in probabilities.items()}
    total = sum(clipped.values())
    if total <= 0:
        return {mode: 1.0 if mode == fallback_mode else 0.0 for mode in clipped}
    return {mode: probability / total for mode, probability in clipped.items()}


def _round_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    return {mode: round(probability, 4) for mode, probability in probabilities.items()}


def _sample_from_distribution(probabilities: dict[str, float]) -> str:
    threshold = random.random()
    cumulative = 0.0
    last_mode = next(iter(probabilities))
    for mode, probability in probabilities.items():
        if probability <= 0:
            continue
        last_mode = mode
        cumulative += probability
        if threshold < cumulative:
            return mode
    return last_mode


def _sample_empty_cell(
    archive: EliteArchive,
    trace: Trace,
    empty_cells: list[tuple[str, int]],
) -> tuple[str, int]:
    if not empty_cells:
        raise ValueError("Cannot initialize an empty cell because the archive has full coverage.")

    if len(archive) == 0:
        return empty_cells[len(trace.hist) % len(empty_cells)]

    occupied_category_counts = {category: 0 for category in archive.categories}
    occupied_bin_counts = {int(depth_bin): 0 for depth_bin in archive.depth_bins}
    for descriptor in archive.occupied_descriptors():
        occupied_category_counts[descriptor.category] = occupied_category_counts.get(descriptor.category, 0) + 1
        occupied_bin_counts[int(descriptor.depth_bin)] = occupied_bin_counts.get(int(descriptor.depth_bin), 0) + 1

    weights = [
        1.0
        / ((1 + occupied_category_counts.get(category, 0)) * (1 + occupied_bin_counts.get(int(depth_bin), 0)))
        for category, depth_bin in empty_cells
    ]
    return random.choices(empty_cells, weights=weights, k=1)[0]


def _format_archive_cell_states(archive: EliteArchive) -> str:
    records = {
        (record.category, int(record.depth_bin)): record
        for record in archive.records()
    }
    lines = []
    for category in archive.categories:
        for depth_bin in archive.depth_bins:
            key = (category, int(depth_bin))
            record = records.get(key)
            if record is None:
                lines.append(f"  - cell=({category}, {depth_bin}), status=empty, quality_metric={DEFAULT_QUALITY_METRIC}, quality=N/A")
            else:
                lines.append(
                    f"  - cell=({category}, {depth_bin}), status=full, "
                    f"quality_metric={DEFAULT_QUALITY_METRIC}, quality={record.quality}"
                )
    return "\n".join(lines)


def _sanitize_elite_record_for_prompt(record, *, include_expression: bool = False) -> dict[str, Any]:
    sanitized = {
        "category": record.category,
        "depth_bin": record.depth_bin,
        "factor_complexity_metric": record.factor_complexity_metric,
        "factor_complexity_value": record.factor_complexity_value,
        "quality_metric": DEFAULT_QUALITY_METRIC,
        "quality": record.quality,
    }
    if include_expression:
        sanitized["factor_expression"] = record.factor_expression
    return sanitized


def _sanitize_search_plan_for_prompt(search_plan: dict[str, Any]) -> dict[str, Any]:
    sanitized = {
        key: value
        for key, value in search_plan.items()
        if key != "parents"
    }
    sanitized_parents = []
    for parent in search_plan.get("parents", []):
        sanitized_parent = {
            "category": parent.get("category"),
            "depth_bin": parent.get("depth_bin"),
            "factor_complexity_metric": parent.get("factor_complexity_metric"),
            "factor_complexity_value": parent.get("factor_complexity_value"),
            "quality_metric": parent.get("quality_metric", DEFAULT_QUALITY_METRIC),
            "quality": parent.get("quality"),
        }
        if parent.get("factor_expression"):
            sanitized_parent["factor_expression"] = parent.get("factor_expression")
        sanitized_parents.append(sanitized_parent)
    sanitized["parents"] = sanitized_parents
    return sanitized


def _format_hypothesis_for_prompt(hypothesis: Hypothesis) -> str:
    lines = [
        f"Hypothesis: {getattr(hypothesis, 'hypothesis', '')}",
        f"Concise Observation: {getattr(hypothesis, 'concise_observation', '')}",
        f"Concise Justification: {getattr(hypothesis, 'concise_justification', '')}",
        f"Concise Knowledge: {getattr(hypothesis, 'concise_knowledge', '')}",
    ]
    concise_specification = getattr(hypothesis, "concise_specification", None)
    if concise_specification is not None:
        lines.append(f"Concise Specification: {concise_specification}")
    return "\n  ".join(lines)


def _format_archive_context(archive: EliteArchive) -> str:
    best = archive.best()
    best_text = "None" if best is None else f"quality={best.quality}, cell=({best.category}, {best.depth_bin})"
    return f"""EliteAlpha MAP-Elites archive:
- Categories: {archive.categories}
- Complexity metric: {archive.complexity_metric_desc()}
- Complexity bins: {archive.depth_bins}
- Quality metric: {DEFAULT_QUALITY_METRIC}
- Coverage: {len(archive)}/{archive.total_cells} = {archive.coverage():.2%}
- QD score: {archive.qd_score()}
- Best elite: {best_text}
- Cell states:
{_format_archive_cell_states(archive)}
"""


def _format_search_plan(search_plan: dict[str, Any]) -> str:
    parents = search_plan.get("parents") or []
    parent_text = "\n".join(
        _format_search_plan_parent(p)
        for p in parents
    ) or "None"
    target_candidate_count = search_plan.get("target_candidate_count")
    target_candidate_count_text = (
        f"- Target candidate count: exactly {target_candidate_count}\n"
        if target_candidate_count
        else ""
    )
    return f"""Current EliteAlpha search plan:
- Mode: {search_plan.get("mode")}
- Mode probabilities: {_format_mode_probabilities(search_plan.get("mode_probabilities"))}
- Archive coverage before selection: {_format_probability(search_plan.get("archive_coverage"))}
- Empty cells before selection: {search_plan.get("empty_cell_count", "unknown")}
- Target category: {search_plan.get("target_category") or "any"}
- Target complexity metric: {search_plan.get("target_complexity_metric")}
- Target complexity bin: {search_plan.get("target_complexity_bin", search_plan.get("target_depth_bin")) or "any"}
{target_candidate_count_text}- User direction: {search_plan.get("potential_direction") or "None"}
- Instruction: {search_plan.get("instruction")}
- Parent elites:
{parent_text}
"""


def _format_search_plan_parent(parent: dict[str, Any]) -> str:
    text = (
        f"- cell=({parent.get('category')}, {parent.get('depth_bin')}), "
        f"metric={parent.get('factor_complexity_metric')}, "
        f"metric_value={parent.get('factor_complexity_value')}, "
        f"quality_metric={parent.get('quality_metric', DEFAULT_QUALITY_METRIC)}, "
        f"quality={parent.get('quality')}"
    )
    expression = parent.get("factor_expression")
    if expression:
        text += f", expression={expression}"
    return text


def _format_mode_probabilities(probabilities: dict[str, float] | None) -> str:
    if not probabilities:
        return "unknown"
    return ", ".join(
        f"{mode}={_format_probability(probability)}"
        for mode, probability in probabilities.items()
    )


def _format_probability(value) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "unknown"


def _format_recent_history(trace: Trace, limit: int = 5) -> str:
    if len(trace.hist) == 0:
        return "No previous hypothesis, experiment, or feedback is available since this is the first round."

    rows = []
    start = max(0, len(trace.hist) - limit)
    for idx, (hypothesis, experiment, feedback) in enumerate(trace.hist[-limit:], start=start):
        implemented = sum(1 for task in experiment.sub_tasks if getattr(task, "factor_implementation", False))
        rows.append(
            f"""Round {idx}:
- Hypothesis: {_format_hypothesis_for_prompt(hypothesis)}
- Candidate count: {len(experiment.sub_tasks)}
- Implemented count: {implemented}
- Result: {experiment.result}
- Feedback observations: {feedback.observations}
- Feedback decision: {feedback.decision}
- Feedback reason: {feedback.reason}
"""
        )
    return "Recent EliteAlpha trace history:\n" + "\n".join(rows)


def _format_history_for_search_plan(trace: Trace, search_plan: dict[str, Any]) -> str:
    if search_plan.get("mode") == "creative_innovation":
        return _format_creative_feedback_history(trace)
    return _format_recent_history(trace)


def _format_creative_feedback_history(
    trace: Trace,
    limit: int = ELITE_ALPHA_CREATIVE_FEEDBACK_HISTORY_LIMIT,
) -> str:
    if len(trace.hist) == 0:
        return (
            "Creative innovation feedback memory:\n"
            "- No previous natural-language feedback is available yet.\n"
            "- Existing factor expressions are not provided."
        )

    start = max(0, len(trace.hist) - limit)
    omitted = len(trace.hist) - start
    rows = []
    if start > 0:
        rows.append(f"(Omitted {start} older rounds to keep the prompt compact.)")

    for idx, (_, _experiment, feedback) in enumerate(trace.hist[-omitted:], start=start):
        rows.append(
            f"""Round {idx} sanitized natural-language feedback:
- Observations: {_sanitize_feedback_text(getattr(feedback, "observations", ""))}
- Hypothesis evaluation: {_sanitize_feedback_text(getattr(feedback, "hypothesis_evaluation", ""))}
- New feedback / direction: {_sanitize_feedback_text(getattr(feedback, "new_hypothesis", ""))}
- Reasoning: {_sanitize_feedback_text(getattr(feedback, "reason", ""))}
- Decision: {getattr(feedback, "decision", None)}
"""
        )

    return (
        "Creative innovation feedback memory:\n"
        "- Use this natural-language feedback only; no existing factor expressions are provided.\n"
        "- Any expression-like fragments in historical feedback have been redacted.\n"
        + "\n".join(rows)
    )


def _sanitize_feedback_text(value: Any) -> str:
    text = "None" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    text = _redact_expression_like_text(text)
    if len(text) > ELITE_ALPHA_FEEDBACK_FIELD_MAX_CHARS:
        text = text[: ELITE_ALPHA_FEEDBACK_FIELD_MAX_CHARS - 3].rstrip() + "..."
    return text or "None"


def _redact_expression_like_text(text: str) -> str:
    text = re.sub(
        r"`[^`\n]*(?:\$[A-Za-z_][A-Za-z0-9_]*|[A-Z_]{2,}\s*\()[^`\n]*`",
        "`<hidden_factor_expression>`",
        text,
    )
    text = re.sub(
        r"\b[A-Z_]{2,}\s*\([^;\n]{0,240}\)",
        "<hidden_factor_expression>",
        text,
    )
    text = re.sub(
        r"\$[A-Za-z_][A-Za-z0-9_]*",
        "<hidden_market_variable>",
        text,
    )
    return text


def _format_archive_targets(archive: EliteArchive) -> str:
    records = archive.to_records()
    if not records:
        return "No elite factors exist yet. Fill the target cell using only the scenario, hypothesis, and archive occupancy signal."
    lines = [
        f"- cell=({record['category']}, {record['depth_bin']}), status=full, "
        f"metric={record.get('factor_complexity_metric')}, metric_value={record.get('factor_complexity_value')}, "
        f"quality_metric={record.get('quality_metric', DEFAULT_QUALITY_METRIC)}, quality={record['quality']}"
        for record in records
    ]
    return "Existing archive occupancy and quality state:\n" + "\n".join(lines)


def _collect_archive_and_history_tasks(trace: Trace) -> list[FactorTask]:
    archive = _require_archive(trace)
    tasks = [record.task for record in archive.records()]
    for _, experiment, _ in trace.hist:
        tasks.extend(experiment.sub_tasks)

    unique_tasks = []
    seen = set()
    for task in tasks:
        if task.factor_name in seen:
            continue
        seen.add(task.factor_name)
        unique_tasks.append(task)
    return unique_tasks


def _elite_hypothesis_specification(base_specification: str) -> str:
    return f"""{base_specification}

  4. **EliteAlpha MAP-Elites Exploration:**
    - Use the provided archive context to decide whether to initialize, mutate, crossover, or creatively innovate factors.
    - Respect the target behavior cell when one is provided; in creative_innovation mode, deliberately spread candidates across any suitable categories and AST depth bins.
    - Prefer hypotheses that can fill empty cells or improve weak occupied cells.
    - Keep novelty relative to parent elites and recent rejected factors.
    - In creative_innovation mode, use only sanitized natural-language feedback and archive cell state; do not infer or reconstruct hidden factor expressions.
"""


def _elite_experiment_output_format(base_output_format: str, archive: EliteArchive) -> str:
    return f"""{base_output_format}

  EliteAlpha extra requirements:
  - Each factor object MUST also include:
    "category": one of {archive.categories}
  - If the current EliteAlpha search plan includes a target candidate count, output that many factor objects in this JSON response.
  - The expression should target the requested complexity bin when possible.
  - Current complexity metric: {archive.complexity_metric_desc()}.
  - Hard constraint: AST depth must be <= {ELITE_ALPHA_MAX_AST_DEPTH}.
  - Do not use a leading negative sign, "-1 * (...)", or "-(...)" only to choose the factor direction. Output the unsigned economic signal; the runner will flip the sign after measuring train Rank IC if needed.
  - Leading sign is ignored when assigning archive complexity, so do not hide a good depth-{ELITE_ALPHA_MAX_AST_DEPTH} idea just because its profitable direction might be negative.
  - Full archive factor expressions are intentionally hidden; when parent expressions are provided in the current search plan, use only those selected parent expressions plus archive cell occupancy and quality signals.
  - In creative_innovation mode, no parent expression is provided; generate several fully new candidates using sanitized natural-language feedback only, and distribute them across any reasonable categories and AST depths.
  - Do not output one-wrapper raw-variable factors such as RANK($open), RANK($return), ZSCORE($open), or ZSCORE($return).
  - Prefer expressions with at least one transformation before cross-sectional ranking/standardization, e.g. combine two variables or use a time-series operator before RANK()/ZSCORE().
  - Archive quality and portfolio selection are ranking-based; the factor does not need to be centered around zero. Avoid algebraically redundant shifts such as `(A / B) - 1` when `A / B` has the same stock ordering and uses less AST depth.
  - If the target complexity bin is too shallow for a non-trivial expression, prioritize passing the acceptance rules over matching that bin exactly.
  - Do not output "ast_depth" unless you are confident; the code will calculate it from the expression.
"""


def _append_expression_rejection_feedback(
    previous: str | None,
    expression: str,
    eval_dict: dict[str, Any],
    *,
    duplication_threshold: int,
    depth_cap: float,
) -> str:
    feedback = _format_expression_rejection_feedback(
        expression,
        eval_dict,
        duplication_threshold=duplication_threshold,
        depth_cap=depth_cap,
    )
    if previous:
        return "\n\n".join([previous, feedback])
    return feedback


def _format_expression_rejection_feedback(
    expression: str,
    eval_dict: dict[str, Any],
    *,
    duplication_threshold: int,
    depth_cap: float,
) -> str:
    reasons = _expression_rejection_reasons(
        eval_dict,
        duplication_threshold=duplication_threshold,
        depth_cap=depth_cap,
    )
    reason_text = "\n".join(f"  - {reason}" for reason in reasons)
    duplicated_subtree = eval_dict.get("duplicated_subtree") or "None"
    return f"""Rejected expression:
- Proposed Expression: {expression}
- Metrics: duplicated_subtree_size={eval_dict.get("duplicated_subtree_size")}, duplicated_subtree={duplicated_subtree}, free_args={eval_dict.get("num_free_args")}, unique_vars={eval_dict.get("num_unique_vars")}, total_nodes={eval_dict.get("num_all_nodes")}, ast_depth={eval_dict.get("ast_depth")}, category={eval_dict.get("factor_category")}
- Rejection reasons:
{reason_text}
- Next attempt guidance:
  - Do not retry one-wrapper raw-variable forms such as RANK($open), RANK($return), ZSCORE($open), or ZSCORE($return).
  - Add structure: combine at least two market variables, or apply a time-series transform before cross-sectional ranking/standardization.
  - Do not spend AST depth on redundant centering such as subtracting 1 from a positive ratio when the ratio itself preserves the same cross-sectional ordering.
  - Keep the expression parseable, economically interpretable, and close to the target category/complexity cell.
"""


def _expression_rejection_reasons(
    eval_dict: dict[str, Any],
    *,
    duplication_threshold: int,
    depth_cap: float,
) -> list[str]:
    reasons = []
    duplicated_size = eval_dict.get("duplicated_subtree_size", 0)
    if duplicated_size > duplication_threshold:
        reasons.append(
            f"duplicated subtree size {duplicated_size} exceeds threshold {duplication_threshold}"
        )

    ast_depth = eval_dict.get("ast_depth", 0)
    if depth_cap != float("inf") and ast_depth > depth_cap:
        reasons.append(f"AST depth {ast_depth} exceeds cap {int(depth_cap)}")

    num_all_nodes = eval_dict.get("num_all_nodes") or 0
    if num_all_nodes <= 0:
        reasons.append("expression has no valid AST nodes")
        return reasons

    num_free_args = eval_dict.get("num_free_args") or 0
    num_unique_vars = eval_dict.get("num_unique_vars") or 0
    free_args_ratio = float(num_free_args) / float(num_all_nodes)
    unique_vars_ratio = float(num_unique_vars) / float(num_all_nodes)

    if free_args_ratio >= 0.5:
        reasons.append(
            f"too many literal/free arguments relative to expression size: {free_args_ratio:.2f} >= 0.50"
        )
    if unique_vars_ratio >= 0.5:
        reasons.append(
            f"expression is too shallow/raw-variable-heavy: unique_vars/total_nodes={unique_vars_ratio:.2f} >= 0.50"
        )

    if not reasons:
        reasons.append("failed the originality/complexity acceptance rules")
    return reasons


def _filter_duplicate_tasks(tasks: list[FactorTask], based_experiments: list[FactorExperiment]) -> list[FactorTask]:
    unique_tasks = []
    for task in tasks:
        duplicate = False
        for based_exp in based_experiments:
            for sub_task in based_exp.sub_tasks:
                if task.factor_name == sub_task.factor_name:
                    duplicate = True
                    break
            if duplicate:
                break
        if not duplicate:
            unique_tasks.append(task)
    return unique_tasks


def _sync_regulator_factor_categories(regulator: FactorRegulator, trace: Trace) -> None:
    zoo = getattr(regulator, "alphazoo", None)
    if zoo is None or zoo.empty:
        return
    if "factor_category" not in zoo.columns:
        zoo["factor_category"] = None

    by_name_and_expr = {}
    by_expr = {}
    for task in _collect_archive_and_history_tasks(trace):
        category = getattr(task, "factor_category", None) or getattr(task, "elite_category", None)
        expression = getattr(task, "factor_expression", None)
        if not category or not expression:
            continue
        by_name_and_expr[(getattr(task, "factor_name", None), expression)] = category
        by_expr.setdefault(expression, category)

    for index, row in zoo.iterrows():
        current = row.get("factor_category")
        if current is not None and str(current).strip() and str(current).lower() != "nan":
            continue
        expression = row.get("factor_expression")
        category = by_name_and_expr.get((row.get("factor_name"), expression)) or by_expr.get(expression)
        if category:
            zoo.at[index, "factor_category"] = category

    regulator.alphazoo = zoo


def _resolve_category(
    archive: EliteArchive,
    raw_category: Any,
    fallback_category: str | None,
    *texts: str,
) -> str:
    candidates = [raw_category, fallback_category, _infer_category_from_text(*texts), archive.categories[0]]
    for candidate in candidates:
        if candidate is None:
            continue
        category = archive.normalize_category(str(candidate))
        if category in archive.categories:
            return category
    return archive.categories[0]


def _resolve_expression_complexity(
    archive: EliteArchive,
    raw_depth: Any,
    expression: str,
    fallback_depth_bin: int | None,
) -> int:
    expression_for_complexity = _expression_without_leading_sign(expression)
    if archive.complexity_metric == "vertex":
        ast_node_count = _expression_ast_node_count(expression_for_complexity)
        if ast_node_count is not None:
            return ast_node_count
        return int(fallback_depth_bin) if fallback_depth_bin is not None else 1

    if raw_depth is not None:
        try:
            return int(raw_depth)
        except (TypeError, ValueError):
            pass

    ast_depth = _expression_ast_depth(expression_for_complexity)
    if ast_depth is not None:
        return ast_depth
    return int(fallback_depth_bin) if fallback_depth_bin is not None else 1


def _expression_without_leading_sign(expression: str) -> str:
    expression = str(expression).strip()
    if expression.startswith("-1 * (") and expression.endswith(")"):
        return expression[len("-1 * (") : -1].strip()
    if expression.startswith("-1*(") and expression.endswith(")"):
        return expression[len("-1*(") : -1].strip()
    if expression.startswith("-1 * "):
        return expression[len("-1 * ") :].strip()
    if expression.startswith("-1*"):
        return expression[len("-1*") :].strip()
    if expression.startswith("-(") and expression.endswith(")"):
        return expression[2:-1].strip()
    if expression.startswith("-"):
        return expression[1:].strip()
    return expression


def _infer_category_from_text(*texts: str) -> str | None:
    text = " ".join(str(t) for t in texts if t).lower()
    if any(key in text for key in ("volume", "turnover", "liquidity", "vwap", "$volume")):
        return "volume-price"
    if any(key in text for key in ("reversal", "mean reversion", "contrarian", "overreaction")):
        return "reversal"
    if any(key in text for key in ("volatility", "variance", "std", "range", "$high", "$low")):
        return "volatility"
    if any(key in text for key in ("rank", "zscore", "cross-sectional", "relative valuation")):
        return "cross-sectional"
    if any(key in text for key in ("momentum", "trend", "return", "delta", "pctchange", "macd", "rsi")):
        return "momentum"
    return None


def _expression_ast_depth(expression: str) -> int | None:
    parsed = _parse_expression(expression)
    if parsed is None:
        return None
    return _node_depth(parsed)


def _expression_ast_node_count(expression: str) -> int | None:
    parsed = _parse_expression(expression)
    if parsed is None:
        return None
    return _node_count(parsed)


def _parse_expression(expression: str) -> Node | None:
    try:
        return parse_expression(expression)
    except Exception as e:
        logger.warning(f"Failed to calculate AST complexity for expression {expression}: {e}")
        return None


def _node_depth(node: Node) -> int:
    if isinstance(node, FunctionNode):
        if not node.args:
            return 1
        return 1 + max(_node_depth(arg) for arg in node.args)
    if isinstance(node, BinaryOpNode):
        return 1 + max(_node_depth(node.left), _node_depth(node.right))
    if isinstance(node, ConditionalNode):
        return 1 + max(
            _node_depth(node.condition),
            _node_depth(node.true_expr),
            _node_depth(node.false_expr),
        )
    return 1


def _node_count(node: Node | None) -> int:
    if node is None:
        return 0
    if isinstance(node, FunctionNode):
        return 1 + sum(_node_count(arg) for arg in node.args)
    if isinstance(node, BinaryOpNode):
        return 1 + _node_count(node.left) + _node_count(node.right)
    if isinstance(node, ConditionalNode):
        return 1 + _node_count(node.condition) + _node_count(node.true_expr) + _node_count(node.false_expr)
    return 1
