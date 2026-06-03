import json
from pathlib import Path

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
from alphaagent.scenarios.qlib.archive import get_task_descriptor, get_task_quality
from alphaagent.utils import convert2bool

rdagent_feedback_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts_rdagent.yaml")
DIRNAME = Path(__file__).absolute().resolve().parent


def process_results(current_result, sota_result):
    # Convert results to aligned Series. The first AlphaAgent round has no
    # accepted SOTA yet, so sota_result can be None.
    current_df = _result_to_series(current_result).to_frame("Current Result")
    sota_df = _result_to_series(sota_result).to_frame("SOTA Result")

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
        combined_result = process_results(exp.result, sota_result)

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


def _format_elite_archive_feedback_context(archive) -> str:
    best = archive.best()
    best_text = "None" if best is None else f"{best.factor_name}, quality={best.quality}, cell=({best.category}, {best.depth_bin})"
    lines = [
        "MAP-Elites archive before this update:",
        f"- Coverage: {len(archive)}/{archive.total_cells} = {archive.coverage():.2%}",
        f"- QD score: {archive.qd_score()}",
        f"- Best elite: {best_text}",
    ]
    records = archive.to_records()
    if records:
        lines.append("- Occupied cells:")
        for record in records:
            lines.append(
                f"  - ({record['category']}, {record['depth_bin']}): "
                f"{record['factor_name']}, quality={record['quality']}, expression={record['factor_expression']}"
            )
    else:
        lines.append("- Occupied cells: None")
    return "\n".join(lines)


def _format_elite_candidate_feedback_context(archive, exp: Experiment) -> tuple[str, bool]:
    lines = ["Candidate archive placements from the current experiment:"]
    any_accepted = False
    for task in exp.sub_tasks:
        descriptor = get_task_descriptor(archive, task)
        quality = get_task_quality(exp, task)
        if descriptor is None:
            lines.append(f"- {task.factor_name}: missing category or AST-depth descriptor; cannot update archive.")
            continue
        if quality is None:
            lines.append(f"- {task.factor_name}: cell=({descriptor.category}, {descriptor.depth_bin}), missing quality metric.")
            continue

        incumbent = archive.get(descriptor)
        accepted = incumbent is None or quality > incumbent.quality
        any_accepted = any_accepted or accepted
        incumbent_text = "empty" if incumbent is None else f"{incumbent.factor_name}, quality={incumbent.quality}"
        lines.append(
            f"- {task.factor_name}: cell=({descriptor.category}, {descriptor.depth_bin}), "
            f"quality={quality}, incumbent={incumbent_text}, archive_acceptance={'yes' if accepted else 'no'}"
        )
    return "\n".join(lines), any_accepted


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
