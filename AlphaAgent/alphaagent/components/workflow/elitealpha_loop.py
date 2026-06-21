"""
EliteAlpha workflow with session control.

This file mirrors AlphaAgentLoop on purpose. EliteAlpha runs the same mining
steps, but its trace owns a MAP-Elites archive that is updated after feedback.
"""

from functools import wraps
from typing import Any
import os
import threading

from alphaagent.components.workflow.conf import BaseFacSetting
from alphaagent.core.developer import Developer
from alphaagent.core.exception import FactorEmptyError
from alphaagent.core.proposal import (
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,
)
from alphaagent.core.scenario import Scenario
from alphaagent.core.utils import import_class
from alphaagent.log import logger
from alphaagent.log.time import measure_time
from alphaagent.scenarios.qlib.archive import (
    DEFAULT_COMPLEXITY_METRIC,
    EliteArchive,
    format_archive_view,
    rebuild_archive_from_trace_history,
    update_archive_from_experiment,
)
from alphaagent.utils.workflow import LoopBase, LoopMeta


STOP_EVENT = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _archive_record_count(archive) -> int | str:
    try:
        return len(archive)
    except Exception:
        return "unknown"


def _archive_is_usable(archive) -> bool:
    return archive is not None and (hasattr(archive, "records") or hasattr(archive, "cells"))


def stop_event_check(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if STOP_EVENT is not None and STOP_EVENT.is_set():
            raise Exception("Operation stopped due to stop_event flag.")
        return func(self, *args, **kwargs)

    return wrapper


class EliteAlphaLoop(LoopBase, metaclass=LoopMeta):
    skip_loop_error = (FactorEmptyError,)

    @measure_time
    def __init__(
        self,
        PROP_SETTING: BaseFacSetting,
        potential_direction,
        stop_event: threading.Event,
        use_local: bool = True,
    ):
        with logger.tag("init"):
            self.use_local = use_local
            logger.info(f"Initialize EliteAlphaLoop, use {'local' if use_local else 'Docker'} backtest")

            scen: Scenario = import_class(PROP_SETTING.scen)(use_local=use_local)
            logger.log_object(scen, tag="scenario")

            self.hypothesis_generator: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(
                scen,
                potential_direction,
            )
            logger.log_object(self.hypothesis_generator, tag="hypothesis generator")

            self.factor_constructor: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()
            logger.log_object(self.factor_constructor, tag="experiment generation")

            self.coder: Developer = import_class(PROP_SETTING.coder)(scen)
            logger.log_object(self.coder, tag="coder")

            self.runner: Developer = import_class(PROP_SETTING.runner)(scen)
            logger.log_object(self.runner, tag="runner")

            self.summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
            logger.log_object(self.summarizer, tag="summarizer")

            trace_cls = import_class(getattr(PROP_SETTING, "trace", "alphaagent.scenarios.qlib.archive.EliteAlphaTrace"))
            archive_kwargs = {
                "complexity_metric": getattr(PROP_SETTING, "archive_complexity_metric", DEFAULT_COMPLEXITY_METRIC)
            }
            archive_vertex_count_thresholds = getattr(PROP_SETTING, "archive_vertex_count_thresholds", None)
            if archive_vertex_count_thresholds is not None:
                archive_kwargs["vertex_count_thresholds"] = archive_vertex_count_thresholds
            archive = EliteArchive(**archive_kwargs)
            self.trace = trace_cls(scen=scen, archive=archive)
            if not hasattr(self.trace, "archive"):
                raise TypeError("EliteAlphaLoop requires a trace with an archive, such as EliteAlphaTrace.")
            logger.info(f"EliteAlpha archive complexity metric: {self.trace.archive.complexity_metric_desc()}")
            logger.log_object(self.trace, tag="elite trace")

            global STOP_EVENT
            STOP_EVENT = stop_event
            super().__init__()

    @classmethod
    def load(cls, path, use_local: bool = True):
        """Load an existing EliteAlpha session."""
        global STOP_EVENT
        STOP_EVENT = None
        instance = super().load(path)
        instance.use_local = use_local
        if not hasattr(instance.trace, "archive"):
            raise TypeError("Loaded session is not an EliteAlpha session: trace has no archive.")
        archive = getattr(instance.trace, "archive", None)
        force_rebuild = _env_flag("ELITEALPHA_FORCE_REBUILD_ARCHIVE")
        if force_rebuild or not _archive_is_usable(archive):
            if not force_rebuild:
                logger.warning("Checkpoint archive is missing or unusable; rebuilding from trace history.")
            rebuild_archive_from_trace_history(instance.trace, log=logger)
        else:
            archive_size = _archive_record_count(archive)
            logger.info(
                "Skip EliteAlpha archive rebuild on resume; "
                f"using checkpoint archive directly, records={archive_size}. "
                "Set ELITEALPHA_FORCE_REBUILD_ARCHIVE=true to rebuild from trace history."
            )
        logger.info(f"Load EliteAlphaLoop, use {'local' if use_local else 'Docker'} backtest")
        return instance

    @measure_time
    @stop_event_check
    def factor_propose(self, prev_out: dict[str, Any]):
        """
        Propose the hypothesis used to construct candidate factors.
        """
        with logger.tag("r"):
            idea = self.hypothesis_generator.gen(self.trace)
            logger.log_object(idea, tag="hypothesis generation")
        return idea

    @measure_time
    @stop_event_check
    def factor_construct(self, prev_out: dict[str, Any]):
        """
        Construct factors from the proposed hypothesis.
        """
        with logger.tag("r"):
            factor = self.factor_constructor.convert(prev_out["factor_propose"], self.trace)
            logger.log_object(factor.sub_tasks, tag="experiment generation")
        return factor

    @measure_time
    @stop_event_check
    def factor_calculate(self, prev_out: dict[str, Any]):
        """
        Calculate factor values from factor expressions.
        """
        with logger.tag("d"):
            factor = self.coder.develop(prev_out["factor_construct"])
            logger.log_object(factor.sub_workspace_list, tag="coder result")
        return factor

    @measure_time
    @stop_event_check
    def factor_backtest(self, prev_out: dict[str, Any]):
        """
        Backtest calculated factors.
        """
        with logger.tag("ef"):
            logger.info(f"Start factor backtest (Local: {self.use_local})")
            if hasattr(self.trace, "archive"):
                setattr(self.runner, "current_elite_archive", self.trace.archive)
            exp = self.runner.develop(prev_out["factor_calculate"], use_local=self.use_local)
            if exp is None:
                logger.error("Factor extraction failed.")
                raise FactorEmptyError("Factor extraction failed.")
            logger.log_object(exp, tag="runner result")
        return exp

    @measure_time
    @stop_event_check
    def feedback(self, prev_out: dict[str, Any]):
        feedback = self.summarizer.generate_feedback(
            prev_out["factor_backtest"],
            prev_out["factor_propose"],
            self.trace,
        )
        with logger.tag("ef"):
            logger.log_object(feedback, tag="feedback")

        update_archive_from_experiment(self.trace.archive, prev_out["factor_backtest"], log=logger)
        logger.log_object(self.trace.archive.to_records(), tag="elite archive")
        logger.log_object(self.trace.archive.history_records(), tag="elite archive history")
        logger.log_object(format_archive_view(self.trace.archive), tag="elite archive view")
        self.trace.hist.append((prev_out["factor_propose"], prev_out["factor_backtest"], feedback))
