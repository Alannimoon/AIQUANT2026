import json
import random
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
from alphaagent.scenarios.qlib.archive import EliteArchive
from alphaagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from alphaagent.scenarios.qlib.proposal.factor_proposal import AlphaAgentHypothesis
from alphaagent.scenarios.qlib.regulator.factor_regulator import FactorRegulator


alphaagent_prompt_dict = Prompts(file_path=Path(__file__).parent.parent / "prompts_alphaagent.yaml")


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
                EliteAlpha Search Plan: {self.elite_search_plan}
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
            _format_recent_history(trace),
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
        response_dict = json.loads(response)
        return EliteAlphaHypothesis(
            hypothesis=response_dict["hypothesis"],
            concise_observation=response_dict["concise_observation"],
            concise_knowledge=response_dict["concise_knowledge"],
            concise_justification=response_dict["concise_justification"],
            concise_specification=response_dict["concise_specification"],
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
        self.factor_regulator = FactorRegulator()
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
                _format_recent_history(trace),
                _format_archive_targets(archive),
            ]
        )

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "function_lib_description": function_lib_description,
            "experiment_output_format": experiment_output_format,
            "target_list": _collect_archive_and_history_tasks(trace),
            "RAG": None,
            "search_plan": search_plan,
        }, True

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
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
        expression_duplication_prompt = None

        for attempt_idx in range(self.max_regeneration_attempts):
            response = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)
            response_dict = json.loads(response)
            attempt_accepted = {}

            for factor_name, factor_info in response_dict.items():
                expr = factor_info["expression"]

                if not self.factor_regulator.is_parsable(expr):
                    logger.info(f"Skip unparsable EliteAlpha expr from {factor_name}: {expr}")
                    continue

                success, eval_dict = self.factor_regulator.evaluate(expr)
                if not success:
                    logger.info(f"Skip unevaluable EliteAlpha expr from {factor_name}: {expr}")
                    continue

                if not self.factor_regulator.is_expression_acceptable(eval_dict):
                    logger.info(
                        "Skip unacceptable EliteAlpha expr from "
                        f"{factor_name}: {expr}; eval={eval_dict}"
                    )
                    expression_duplication_prompt = _append_duplication_feedback(
                        expression_duplication_prompt,
                        expr,
                        eval_dict,
                    )
                    context["expression_duplication"] = expression_duplication_prompt
                    continue

                attempt_accepted[factor_name] = factor_info

            if attempt_accepted:
                accepted_response_dict.update(attempt_accepted)
                break

            logger.info(
                f"No acceptable EliteAlpha factor expressions in attempt "
                f"{attempt_idx + 1}/{self.max_regeneration_attempts}; retrying..."
            )
            if expression_duplication_prompt is not None:
                user_prompt = self._render_user_prompt(context)

        if not accepted_response_dict:
            raise ValueError("Failed to generate acceptable EliteAlpha factor expressions.")

        proposed_names = list(accepted_response_dict)
        proposed_exprs = [factor_info["expression"] for factor_info in accepted_response_dict.values()]
        self.factor_regulator.add_factor(proposed_names, proposed_exprs)
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
                expression_duplication=context.get("expression_duplication"),
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
    if len(archive) == 0:
        category = archive.categories[len(trace.hist) % len(archive.categories)]
        depth_bin = archive.depth_bins[len(trace.hist) % len(archive.depth_bins)]
        return {
            "mode": "initialize",
            "target_category": category,
            "target_depth_bin": depth_bin,
            "target_complexity_bin": depth_bin,
            "target_complexity_metric": archive.complexity_metric,
            "parents": [],
            "instruction": "Seed an empty MAP-Elites archive with a diverse, testable factor.",
            "potential_direction": potential_direction,
        }

    if len(archive) >= 2 and random.random() < 0.3:
        left, right = archive.sample_pair(weighted=True)
        target_category = left.category if random.random() < 0.5 else right.category
        target_depth_bin = random.choice((left.depth_bin, right.depth_bin))
        return {
            "mode": "crossover",
            "target_category": target_category,
            "target_depth_bin": target_depth_bin,
            "target_complexity_bin": target_depth_bin,
            "target_complexity_metric": archive.complexity_metric,
            "parents": [left.to_dict(), right.to_dict()],
            "instruction": "Combine useful ideas from the two parent elites without copying either expression.",
            "potential_direction": potential_direction,
        }

    parent = archive.sample_parent(weighted=True)
    return {
        "mode": "mutation",
        "target_category": parent.category,
        "target_depth_bin": parent.depth_bin,
        "target_complexity_bin": parent.depth_bin,
        "target_complexity_metric": archive.complexity_metric,
        "parents": [parent.to_dict()],
        "instruction": "Mutate the parent elite while preserving its broad behavioral intent and improving originality.",
        "potential_direction": potential_direction,
    }


def _format_archive_context(archive: EliteArchive) -> str:
    best = archive.best()
    best_text = "None" if best is None else f"{best.factor_name}, quality={best.quality}, cell=({best.category}, {best.depth_bin})"
    occupied = ", ".join(f"({d.category}, {d.depth_bin})" for d in archive.occupied_descriptors()) or "None"
    return f"""EliteAlpha MAP-Elites archive:
- Categories: {archive.categories}
- Complexity metric: {archive.complexity_metric_desc()}
- Complexity bins: {archive.depth_bins}
- Coverage: {len(archive)}/{archive.total_cells} = {archive.coverage():.2%}
- QD score: {archive.qd_score()}
- Best elite: {best_text}
- Occupied cells: {occupied}
"""


def _format_search_plan(search_plan: dict[str, Any]) -> str:
    parents = search_plan.get("parents") or []
    parent_text = "\n".join(
        f"- {p.get('factor_name')}: category={p.get('category')}, complexity_bin={p.get('depth_bin')}, "
        f"metric={p.get('factor_complexity_metric')}, metric_value={p.get('factor_complexity_value')}, "
        f"quality={p.get('quality')}, expression={p.get('factor_expression')}"
        for p in parents
    ) or "None"
    return f"""Current EliteAlpha search plan:
- Mode: {search_plan.get("mode")}
- Target category: {search_plan.get("target_category")}
- Target complexity metric: {search_plan.get("target_complexity_metric")}
- Target complexity bin: {search_plan.get("target_complexity_bin", search_plan.get("target_depth_bin"))}
- Instruction: {search_plan.get("instruction")}
- Parent elites:
{parent_text}
"""


def _format_recent_history(trace: Trace, limit: int = 5) -> str:
    if len(trace.hist) == 0:
        return "No previous hypothesis, experiment, or feedback is available since this is the first round."

    rows = []
    start = max(0, len(trace.hist) - limit)
    for idx, (hypothesis, experiment, feedback) in enumerate(trace.hist[-limit:], start=start):
        factor_names = [task.factor_name for task in experiment.sub_tasks]
        rows.append(
            f"""Round {idx}:
- Hypothesis: {hypothesis}
- Factors: {factor_names}
- Result: {experiment.result}
- Feedback observations: {feedback.observations}
- Feedback decision: {feedback.decision}
- Feedback reason: {feedback.reason}
"""
        )
    return "Recent EliteAlpha trace history:\n" + "\n".join(rows)


def _format_archive_targets(archive: EliteArchive) -> str:
    records = archive.to_records()
    if not records:
        return "No elite factors exist yet. Avoid obvious textbook factors and fill the target cell first."
    lines = [
        f"- {record['factor_name']}: cell=({record['category']}, {record['depth_bin']}), "
        f"metric={record.get('factor_complexity_metric')}, metric_value={record.get('factor_complexity_value')}, "
        f"quality={record['quality']}, expression={record['factor_expression']}"
        for record in records
    ]
    return "Existing elite factors to avoid directly duplicating:\n" + "\n".join(lines)


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
    - Use the provided archive context to decide whether to initialize, mutate, or crossover factors.
    - Respect the target behavior cell: category and AST depth bin.
    - Prefer hypotheses that can fill empty cells or improve weak occupied cells.
    - Keep novelty relative to parent elites and recent rejected factors.
"""


def _elite_experiment_output_format(base_output_format: str, archive: EliteArchive) -> str:
    return f"""{base_output_format}

  EliteAlpha extra requirements:
  - Each factor object MUST also include:
    "category": one of {archive.categories}
  - The expression should target the requested complexity bin when possible.
  - Current complexity metric: {archive.complexity_metric_desc()}.
  - Do not output "ast_depth" unless you are confident; the code will calculate it from the expression.
"""


def _append_duplication_feedback(previous: str | None, expression: str, eval_dict: dict[str, Any]) -> str:
    feedback = (
        Environment(undefined=StrictUndefined)
        .from_string(alphaagent_prompt_dict["expression_duplication"])
        .render(
            prev_expression=expression,
            duplicated_subtree_size=eval_dict["duplicated_subtree_size"],
            duplicated_subtree=eval_dict["duplicated_subtree"],
        )
    )
    if previous:
        return "\n\n".join([previous, feedback])
    return feedback


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
    if archive.complexity_metric == "vertex":
        ast_node_count = _expression_ast_node_count(expression)
        if ast_node_count is not None:
            return ast_node_count
        return int(fallback_depth_bin) if fallback_depth_bin is not None else 1

    if raw_depth is not None:
        try:
            return int(raw_depth)
        except (TypeError, ValueError):
            pass

    ast_depth = _expression_ast_depth(expression)
    if ast_depth is not None:
        return ast_depth
    return int(fallback_depth_bin) if fallback_depth_bin is not None else 1


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
