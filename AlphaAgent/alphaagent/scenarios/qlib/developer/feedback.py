import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, StrictUndefined

from alphaagent.core.experiment import Experiment
from alphaagent.core.prompts import Prompts
from alphaagent.core.proposal import (
    Hypothesis,
    HypothesisExperiment2Feedback,
    HypothesisFeedback,
    Trace,
)
from alphaagent.log import logger
from alphaagent.oai.llm_utils import APIBackend
from alphaagent.scenarios.qlib.archive import DEFAULT_QUALITY_METRIC, get_task_descriptor, get_task_quality
from alphaagent.utils import convert2bool

rdagent_feedback_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts_rdagent.yaml")
DIRNAME = Path(__file__).absolute().resolve().parent


def process_results(current_result, sota_result):
    # Convert Series to DataFrame with named columns. On the very first loop
    # `sota_result` can come through as a scalar (no prior SOTA yet) which the
    # dict-of-scalar form of pd.DataFrame rejects; in that case, build an empty
    # SOTA column indexed by the current metrics so the downstream `concat`
    # still works.
    current_df = pd.DataFrame({"Current Result": current_result})
    if hasattr(sota_result, "index"):
        sota_df = pd.DataFrame({"SOTA Result": sota_result})
    else:
        # First loop: no SOTA yet. Fill with -inf so the downstream lambda
        # `Current > SOTA` always picks Current (we have no benchmark to beat).
        sota_df = pd.DataFrame(
            {"SOTA Result": [float("-inf")] * len(current_df)},
            index=current_df.index,
        )

    current_df.index.name = "metric"
    sota_df.index.name = "metric"

    # Combine the dataframes on the Metric index
    combined_df = pd.concat([current_df, sota_df], axis=1)

    # Select important metrics for comparison
    important_metrics = [
        "1day.excess_return_without_cost.max_drawdown",
        "1day.excess_return_without_cost.information_ratio",
        "1day.excess_return_without_cost.annualized_return",
        "IC",
    ]

    # Filter to only available metrics (skip missing ones like portfolio metrics)
    available_metrics = [m for m in important_metrics if m in combined_df.index]
    filtered_combined_df = combined_df.loc[available_metrics]

    if "SOTA Result" not in filtered_combined_df:
        filtered_combined_df["SOTA Result"] = pd.NA

    filtered_combined_df[
        "Bigger columns name (Didn't consider the direction of the metric, you should judge it by yourself that bigger is better or smaller is better)"
    ] = filtered_combined_df.apply(_compare_current_and_sota, axis=1)

    return filtered_combined_df.to_string()


def _result_to_series(result) -> pd.Series:
    if result is None:
        return pd.Series(dtype="object")
    if isinstance(result, pd.Series):
        return result
    if isinstance(result, pd.DataFrame):
        if result.shape[1] == 1:
            return result.iloc[:, 0]
        return result.stack()
    if isinstance(result, dict):
        return pd.Series(result)
    if isinstance(result, (list, tuple)):
        return pd.Series(result)
    return pd.Series({"value": result})


def _compare_current_and_sota(row: pd.Series) -> str:
    current = pd.to_numeric(row.get("Current Result"), errors="coerce")
    sota = pd.to_numeric(row.get("SOTA Result"), errors="coerce")
    if pd.isna(current) and pd.isna(sota):
        return "N/A"
    if pd.isna(sota):
        return "Current Result (no SOTA result yet)"
    if pd.isna(current):
        return "SOTA Result"
    return "Current Result" if current > sota else "SOTA Result"


def combined_result_new(exp: Experiment, sota_result=None, archive=None) -> str:
    """
    Build compact EliteAlpha feedback.

    The archive section intentionally hides incumbent factor names and
    expressions. The LLM sees cell-level quality and replacement pressure, not a
    leaderboard of exact factors to overfit.
    """
    sections = [
        "EliteAlpha Combined Result",
        "",
        _format_candidate_factor_metrics(exp, archive),
        "",
        _format_experiment_metric_comparison(exp.result, sota_result),
    ]
    if archive is not None:
        sections.extend(
            [
                "",
                _format_archive_grid_state(archive),
                "",
                _format_next_round_guidance(exp, archive),
            ]
        )
    return "\n".join(section for section in sections if section is not None)


def _format_candidate_factor_metrics(exp: Experiment, archive) -> str:
    rows = []
    for task in exp.sub_tasks:
        factor_name = getattr(task, "factor_name", getattr(task, "name", "unknown_factor"))
        sub_result = _get_factor_sub_result(exp, factor_name)
        descriptor = get_task_descriptor(archive, task) if archive is not None else None
        incumbent = archive.get(descriptor) if archive is not None and descriptor is not None else None
        quality = get_task_quality(exp, task)
        incumbent_quality = getattr(incumbent, "quality", None)

        rows.append(
            {
                "factor": factor_name,
                "implemented": getattr(task, "factor_implementation", None),
                "cell": _format_cell(descriptor),
                "IC": _lookup_metric(sub_result, _IC_KEYS),
                "Rank IC": _lookup_metric(sub_result, _RANK_IC_KEYS),
                "ICIR": _lookup_metric(sub_result, _ICIR_KEYS),
                "Rank ICIR": _lookup_metric(sub_result, _RANK_ICIR_KEYS),
                "annualized_return": _lookup_metric(sub_result, _ANNUALIZED_RETURN_KEYS),
                "information_ratio": _lookup_metric(sub_result, _INFORMATION_RATIO_KEYS),
                "max_drawdown": _lookup_metric(sub_result, _MAX_DRAWDOWN_KEYS),
                "archive_quality_metric": DEFAULT_QUALITY_METRIC,
                "cell_sota_quality": incumbent_quality,
                "quality_delta_vs_cell_sota": _numeric_delta(quality, incumbent_quality),
                "archive_action": _candidate_archive_action(quality, incumbent),
                "sign_flipped": _lookup_metric(sub_result, ("sign_flipped", "Sign Flipped")),
            }
        )

    if not rows:
        return "Candidate single-factor metrics:\n(no candidate factors)"

    df = pd.DataFrame(rows)
    text = "Candidate single-factor metrics and cell-SOTA comparison:\n"
    text += _format_table(df)

    if df[["annualized_return", "information_ratio", "max_drawdown"]].isna().all().all():
        text += (
            "\nNote: single-factor portfolio annualized_return/information_ratio/max_drawdown "
            "are not available in the current runner output; use the experiment-level portfolio "
            "comparison below for those portfolio metrics."
        )
    return text


def _format_experiment_metric_comparison(current_result, sota_result) -> str:
    rows = []
    for label, keys, preference in _EXPERIMENT_METRICS:
        current = _lookup_result_metric(current_result, keys)
        sota = _lookup_result_metric(sota_result, keys)
        rows.append(
            {
                "metric": label,
                "current": current,
                "sota": sota,
                "delta": _numeric_delta(current, sota),
                "winner": _compare_metric_values(current, sota, preference),
            }
        )
    return "Experiment-level portfolio/signal metrics vs SOTA:\n" + _format_table(pd.DataFrame(rows))


def _format_archive_grid_state(archive) -> str:
    records = {
        (record.category, int(record.depth_bin)): record
        for record in archive.records()
    }
    replacement_counts = _archive_replacement_counts(archive)

    rows = []
    for category in archive.categories:
        for depth_bin in archive.depth_bins:
            key = (category, int(depth_bin))
            record = records.get(key)
            rows.append(
                {
                    "cell": f"({category}, {depth_bin})",
                    "occupied": record is not None,
                    "quality_metric": DEFAULT_QUALITY_METRIC,
                    "quality_score": None if record is None else record.quality,
                    "replacement_count": replacement_counts.get(key, 0),
                }
            )

    header = [
        "Archive state before current update (sanitized):",
        f"- Coverage: {len(archive)}/{archive.total_cells} = {archive.coverage():.2%}",
        f"- QD score: {_format_number(archive.qd_score())}",
        "- Cell details: factor names and expressions are intentionally hidden.",
        f"- Quality metric: {DEFAULT_QUALITY_METRIC}.",
    ]
    return "\n".join(header) + "\n" + _format_table(pd.DataFrame(rows))


def _format_next_round_guidance(exp: Experiment, archive) -> str:
    records = {
        (record.category, int(record.depth_bin)): record
        for record in archive.records()
    }
    empty_cells = [
        (category, int(depth_bin))
        for category in archive.categories
        for depth_bin in archive.depth_bins
        if (category, int(depth_bin)) not in records
    ]
    weak_cells = sorted(
        [
            (category, depth_bin, record.quality)
            for (category, depth_bin), record in records.items()
        ],
        key=lambda item: item[2],
    )[:5]

    candidate_rows = []
    accepted_count = 0
    for task in exp.sub_tasks:
        descriptor = get_task_descriptor(archive, task)
        quality = get_task_quality(exp, task)
        incumbent = archive.get(descriptor) if descriptor is not None else None
        action = _candidate_archive_action(quality, incumbent)
        accepted_count += int(action in {"fill_empty_cell", "replace_incumbent"})
        candidate_rows.append(f"{getattr(task, 'factor_name', 'unknown_factor')}: {action}")

    lines = ["Next-round guidance summary:"]
    if empty_cells:
        lines.append(
            f"- Prioritize coverage: {len(empty_cells)} empty cells remain; examples: "
            f"{_format_cell_examples(empty_cells)}."
        )
    elif weak_cells:
        lines.append(
            "- Archive is fully covered; prioritize weak-cell improvement over tiny edits to the global best."
        )

    if weak_cells:
        weak_text = ", ".join(
            f"({category}, {depth_bin}) q={_format_number(quality)}"
            for category, depth_bin, quality in weak_cells
        )
        lines.append(f"- Weak occupied cells to improve: {weak_text}.")

    if candidate_rows:
        lines.append(f"- Current candidate archive actions: {'; '.join(candidate_rows)}.")
        if accepted_count == 0:
            lines.append(
                "- No candidate is expected to improve the archive; change the economic mechanism, "
                "normalization, or target cell rather than only changing one window size."
            )
    lines.append(
        "- Keep diversity pressure: use mutation/crossover only when it changes the expression structure "
        "and targets an empty or weak cell."
    )
    return "\n".join(lines)


def _format_elite_archive_feedback_context(archive) -> str:
    return _format_archive_grid_state(archive)


def _format_elite_candidate_feedback_context(archive, exp: Experiment) -> tuple[str, bool]:
    lines = ["Candidate archive placements from the current experiment:"]
    any_accepted = False
    for task in exp.sub_tasks:
        descriptor = get_task_descriptor(archive, task)
        quality = get_task_quality(exp, task)
        if descriptor is None:
            lines.append(f"- {task.factor_name}: missing category or complexity descriptor; cannot update archive.")
            continue
        if quality is None:
            lines.append(f"- {task.factor_name}: cell=({descriptor.category}, {descriptor.depth_bin}), missing quality metric.")
            continue

        incumbent = archive.get(descriptor)
        incumbent_quality = None if incumbent is None else incumbent.quality
        accepted = incumbent is None or quality > incumbent_quality
        any_accepted = any_accepted or accepted
        lines.append(
            f"- {task.factor_name}: cell=({descriptor.category}, {descriptor.depth_bin}), "
            f"quality={_format_number(quality)}, incumbent_quality={_format_number(incumbent_quality)}, "
            f"quality_delta={_format_number(_numeric_delta(quality, incumbent_quality))}, "
            f"archive_action={_candidate_archive_action(quality, incumbent)}"
        )
    return "\n".join(lines), any_accepted


_IC_KEYS = ("IC", "ic")
_RANK_IC_KEYS = ("Rank IC", "RankIC", "rank_ic", "rank ic")
_ICIR_KEYS = ("ICIR", "IC IR", "icir", "ic_ir")
_RANK_ICIR_KEYS = ("Rank ICIR", "RankICIR", "rank_icir", "rank icir")
_ANNUALIZED_RETURN_KEYS = (
    "1day.excess_return_without_cost.annualized_return",
    "annualized_return",
    "Annualized Return",
)
_INFORMATION_RATIO_KEYS = (
    "1day.excess_return_without_cost.information_ratio",
    "information_ratio",
    "Information Ratio",
    "IR",
)
_MAX_DRAWDOWN_KEYS = (
    "1day.excess_return_without_cost.max_drawdown",
    "max_drawdown",
    "Max Drawdown",
)
_EXPERIMENT_METRICS = (
    ("single/batch IC", _IC_KEYS, "higher"),
    ("single/batch Rank IC", _RANK_IC_KEYS, "higher"),
    ("annualized_return", _ANNUALIZED_RETURN_KEYS, "higher"),
    ("information_ratio", _INFORMATION_RATIO_KEYS, "higher"),
    ("max_drawdown_abs", _MAX_DRAWDOWN_KEYS, "lower_abs"),
)


def _get_factor_sub_result(exp: Experiment, factor_name: str) -> Mapping[str, Any]:
    sub_result = getattr(exp, "sub_results", {}).get(factor_name)
    if isinstance(sub_result, Mapping):
        return sub_result
    if sub_result is None:
        return {}
    return {"quality": sub_result, "Rank IC": sub_result}


def _lookup_metric(values: Mapping[str, Any], keys: tuple[str, ...]):
    if not values:
        return None
    normalized = {_normalize_key(key): key for key in values}
    for key in keys:
        actual_key = normalized.get(_normalize_key(key))
        if actual_key is not None:
            return _coerce_display_value(values[actual_key])
    return None


def _lookup_result_metric(result, keys: tuple[str, ...]):
    if result is None:
        return None
    series = _result_to_series(result)
    if series.empty:
        return None
    normalized = {_normalize_key(key): key for key in series.index}
    for key in keys:
        actual_key = normalized.get(_normalize_key(key))
        if actual_key is not None:
            return _coerce_display_value(series.loc[actual_key])
    return None


def _normalize_key(value: Any) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def _coerce_display_value(value):
    if isinstance(value, (pd.Series, pd.DataFrame)):
        if len(value) == 1:
            value = value.iloc[0]
        else:
            return str(value)
    if isinstance(value, bool):
        return value
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric):
        return float(numeric)
    if value is None or pd.isna(value):
        return None
    return value


def _numeric_delta(current, baseline):
    current_num = pd.to_numeric(current, errors="coerce")
    baseline_num = pd.to_numeric(baseline, errors="coerce")
    if pd.isna(current_num) or pd.isna(baseline_num):
        return None
    return float(current_num - baseline_num)


def _compare_metric_values(current, baseline, preference: str) -> str:
    current_num = pd.to_numeric(current, errors="coerce")
    baseline_num = pd.to_numeric(baseline, errors="coerce")
    if pd.isna(current_num) and pd.isna(baseline_num):
        return "N/A"
    if pd.isna(baseline_num):
        return "current (no SOTA)"
    if pd.isna(current_num):
        return "SOTA"
    if preference == "lower_abs":
        return "current" if abs(current_num) < abs(baseline_num) else "SOTA"
    return "current" if current_num > baseline_num else "SOTA"


def _candidate_archive_action(quality, incumbent) -> str:
    if quality is None:
        return "missing_quality"
    if incumbent is None:
        return "fill_empty_cell"
    return "replace_incumbent" if quality > incumbent.quality else "reject_below_incumbent"


def _archive_replacement_counts(archive) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = {}
    for item in getattr(archive, "hist", []):
        if not getattr(item, "accepted", False) or getattr(item, "incumbent", None) is None:
            continue
        descriptor = getattr(item, "descriptor", None)
        if descriptor is None:
            continue
        key = (descriptor.category, int(descriptor.depth_bin))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _format_cell(descriptor) -> str:
    if descriptor is None:
        return "N/A"
    return f"({descriptor.category}, {descriptor.depth_bin})"


def _format_cell_examples(cells: list[tuple[str, int]], limit: int = 8) -> str:
    examples = ", ".join(f"({category}, {depth_bin})" for category, depth_bin in cells[:limit])
    if len(cells) > limit:
        examples += ", ..."
    return examples


def _format_number(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return str(value)
    return f"{float(numeric):.6g}"


def _format_table(df: pd.DataFrame) -> str:
    display_df = df.copy()
    for column in display_df.columns:
        display_df[column] = display_df[column].map(_format_number)
    return display_df.to_string(index=False)


class QlibFactorHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):
    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibFactorExperiment): The experiment to generate feedback for.
            hypothesis (QlibFactorHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        sota_result = exp.based_experiments[-1].result

        # Process the results to filter important metrics
        combined_result = process_results(current_result, sota_result)

        # Generate the system prompt
        sys_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(rdagent_feedback_prompts["factor_feedback_generation"]["system"])
            .render(scenario=self.scen.get_scenario_all_desc())
        )

        # Generate the user prompt
        usr_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(rdagent_feedback_prompts["factor_feedback_generation"]["user"])
            .render(
                hypothesis_text=hypothesis_text,
                task_details=tasks_factors,
                combined_result=combined_result,
            )
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )

        # Parse the JSON response to extract the feedback
        response_json = json.loads(response)

        # Extract fields from JSON response
        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )



alphaagent_feedback_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts_alphaagent.yaml")
elitealpha_feedback_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts_elitealpha.yaml")


class AlphaAgentQlibFactorHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):
    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibFactorExperiment): The experiment to generate feedback for.
            hypothesis (QlibFactorHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        sota_result = exp.based_experiments[-1].result

        # Process the results to filter important metrics
        combined_result = process_results(current_result, sota_result)

        # Generate the system prompt
        sys_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(alphaagent_feedback_prompts["factor_feedback_generation"]["system"])
            .render(scenario=self.scen.get_scenario_all_desc())
        )

        # Generate the user prompt
        usr_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(alphaagent_feedback_prompts["factor_feedback_generation"]["user"])
            .render(
                hypothesis_text=hypothesis_text,
                task_details=tasks_factors,
                combined_result=combined_result,
            )
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )

        # Parse the JSON response to extract the feedback
        response_json = json.loads(response)

        # Extract fields from JSON response
        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )


class EliteAlphaQlibFactorHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):
    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        logger.info("Generating EliteAlpha feedback...")
        archive = getattr(trace, "archive", None)
        if archive is None:
            raise TypeError("EliteAlpha feedback requires trace.archive. Use EliteAlphaTrace with EliteAlphaLoop.")

        hypothesis_text = hypothesis.hypothesis
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        sota_result = exp.based_experiments[-1].result
        combined_result = combined_result_new(exp, sota_result=sota_result, archive=archive)

        archive_context = _format_elite_archive_feedback_context(archive)
        candidate_context, deterministic_archive_acceptance = _format_elite_candidate_feedback_context(archive, exp)

        base_system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(alphaagent_feedback_prompts["factor_feedback_generation"]["system"])
            .render(scenario=self.scen.get_scenario_all_desc())
        )
        elite_system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(elitealpha_feedback_prompts["factor_feedback_generation"]["system_addendum"])
            .render()
        )
        sys_prompt = f"{base_system_prompt}\n\n{elite_system_prompt}"

        base_user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(alphaagent_feedback_prompts["factor_feedback_generation"]["user"])
            .render(
                hypothesis_text=hypothesis_text,
                task_details=tasks_factors,
                combined_result=combined_result,
            )
        )
        elite_user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(elitealpha_feedback_prompts["factor_feedback_generation"]["user_addendum"])
            .render(
                archive_context=archive_context,
                candidate_context=candidate_context,
                deterministic_archive_acceptance="yes" if deterministic_archive_acceptance else "no",
            )
        )
        usr_prompt = f"{base_user_prompt}\n\n{elite_user_prompt}"

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )
        response_json = json.loads(response)

        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        llm_decision = convert2bool(response_json.get("Replace Best Result", "no"))
        decision = deterministic_archive_acceptance or llm_decision

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )


class QlibModelHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):
    """Generated feedbacks on the hypothesis from **Executed** Implementations of different tasks & their comparisons with previous performances"""

    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        The `ti` should be executed and the results should be included, as well as the comparison between previous results (done by LLM).
        For example: `mlflow` of Qlib will be included.
        """

        logger.info("Generating feedback...")
        # Define the system prompt for hypothesis feedback
        system_prompt = feedback_prompts["model_feedback_generation"]["system"]

        # Define the user prompt for hypothesis feedback
        context = trace.scen
        SOTA_hypothesis, SOTA_experiment = trace.get_sota_hypothesis_and_experiment()

        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(feedback_prompts["model_feedback_generation"]["user"])
            .render(
                context=context,
                last_hypothesis=SOTA_hypothesis,
                last_task=SOTA_experiment.sub_tasks[0].get_task_information() if SOTA_hypothesis else None,
                last_code=SOTA_experiment.sub_workspace_list[0].code_dict.get("model.py") if SOTA_hypothesis else None,
                last_result=SOTA_experiment.result if SOTA_hypothesis else None,
                hypothesis=hypothesis,
                exp=exp,
            )
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response_hypothesis = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
        )

        # Parse the JSON response to extract the feedback
        response_json_hypothesis = json.loads(response_hypothesis)
        return HypothesisFeedback(
            observations=response_json_hypothesis.get("Observations", "No observations provided"),
            hypothesis_evaluation=response_json_hypothesis.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=response_json_hypothesis.get("New Hypothesis", "No new hypothesis provided"),
            reason=response_json_hypothesis.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(response_json_hypothesis.get("Decision", "false")),
        )
