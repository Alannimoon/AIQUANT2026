import pickle
import hashlib
from pathlib import Path
from typing import Any, List
import os
import re
import numpy as np
import pandas as pd
from pandarallel import pandarallel

from alphaagent.core.conf import ExtendedBaseSettings, ExtendedSettingsConfigDict, RD_AGENT_SETTINGS
from alphaagent.core.utils import cache_with_pickle, multiprocessing_wrapper

pandarallel.initialize(verbose=1)

from alphaagent.components.runner import CachedRunner
from alphaagent.core.exception import FactorEmptyError
from alphaagent.log import logger
from alphaagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment

DIRNAME = Path(__file__).absolute().resolve().parent
DIRNAME_local = Path.cwd()
FACTOR_LEVEL_QUALITY_CACHE_VERSION = "paper_direct_ic_train_valid_v1"


class QlibFactorRunnerSettings(ExtendedBaseSettings):
    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_FACTOR_")

    qlib_config_name: str = "conf_cn_combined_kdd_ver.yaml"
    use_lightweight_qlib_test: bool = False
    lightweight_qlib_config_name: str = "conf_cn_combined_light.yaml"
    archive_backtest_streaming: bool = False
    archive_backtest_warmup_days: int = 180

    @property
    def active_qlib_config_name(self) -> str:
        if self.use_lightweight_qlib_test:
            return self.lightweight_qlib_config_name
        return self.qlib_config_name


QLIB_FACTOR_RUNNER_SETTINGS = QlibFactorRunnerSettings()


def get_factor_runner_cache_key(self, exp: QlibFactorExperiment, **kwargs) -> str:
    base_key = CachedRunner.get_cache_key(self, exp, **kwargs)
    return f"{FACTOR_LEVEL_QUALITY_CACHE_VERSION}_{QLIB_FACTOR_RUNNER_SETTINGS.active_qlib_config_name}_{base_key}"

# class QlibFactorExpWorkspace:

#     def prepare():
#         # create a folder;
#         # copy template
#         # place data inside the folder `combined_factors`
#         #
#     def execute():
#         de = DockerEnv()
#         de.run(local_path=self.ws_path, entry="qrun conf.yaml")

# TODO: supporting multiprocessing and keep previous results


class QlibFactorRunner(CachedRunner[QlibFactorExperiment]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - price-volume data dumper
    - `data.py` + Adaptor to Factor implementation
    - results in `mlflow`
    """

    def calculate_information_coefficient(
        self, concat_feature: pd.DataFrame, SOTA_feature_column_size: int, new_feature_columns_size: int
    ) -> pd.DataFrame:
        res = pd.Series(index=range(SOTA_feature_column_size * new_feature_columns_size))
        for col1 in range(SOTA_feature_column_size):
            for col2 in range(SOTA_feature_column_size, SOTA_feature_column_size + new_feature_columns_size):
                res.loc[col1 * new_feature_columns_size + col2 - SOTA_feature_column_size] = concat_feature.iloc[
                    :, col1
                ].corr(concat_feature.iloc[:, col2])
        return res

    def deduplicate_new_factors(self, SOTA_feature: pd.DataFrame, new_feature: pd.DataFrame) -> pd.DataFrame:
        # calculate the IC between each column of SOTA_feature and new_feature
        # if the IC is larger than a threshold, remove the new_feature column
        # return the new_feature

        concat_feature = pd.concat([SOTA_feature, new_feature], axis=1)
        IC_max = (
            concat_feature.groupby("datetime")
            .parallel_apply(
                lambda x: self.calculate_information_coefficient(x, SOTA_feature.shape[1], new_feature.shape[1])
            )
            .mean()
        )
        IC_max.index = pd.MultiIndex.from_product([range(SOTA_feature.shape[1]), range(new_feature.shape[1])])
        IC_max = IC_max.unstack().max(axis=0)
        return new_feature.iloc[:, IC_max[IC_max < 0.99].index]

    @cache_with_pickle(get_factor_runner_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibFactorExperiment, use_local: bool = True) -> QlibFactorExperiment:
        
        """
        Generate the experiment by processing and combining factor data,
        then passing the combined data to Docker or local environment for backtest results.
        """
        config_name = QLIB_FACTOR_RUNNER_SETTINGS.active_qlib_config_name
        logger.info(
            "Use qlib factor config "
            f"{config_name} (lightweight={QLIB_FACTOR_RUNNER_SETTINGS.use_lightweight_qlib_test})"
        )
        if not hasattr(exp.experiment_workspace, "template_folder_path"):
            exp.experiment_workspace.template_folder_path = DIRNAME.parent / "experiment" / "factor_template"
        
        if exp.based_experiments:
            last_base_exp = exp.based_experiments[-1]
            has_base_content = bool(getattr(last_base_exp, "sub_tasks", None)) or bool(
                getattr(last_base_exp, "sub_workspace_list", None)
            )
            if has_base_content and last_base_exp.result is None:
                exp.based_experiments[-1] = self.develop(last_base_exp, use_local=use_local)

        quality_scope = self.load_quality_scope(
            config_name,
            exp.experiment_workspace.template_folder_path,
        )

        should_merge_sota = self.should_merge_sota_factors(exp)
        SOTA_factor = None
        if should_merge_sota and len(exp.based_experiments) > 1:
            try:
                SOTA_factor = self.process_factor_data(exp.based_experiments)
            except FactorEmptyError as e:
                logger.warning(f"SOTA factor merge skipped: failed to load historical factor data: {e}")
        elif not should_merge_sota:
            logger.info("Skip historical SOTA factor merge for EliteAlpha archive scoring.")

        # Process the new factors for every round, including round 1. The
        # archive quality must come from direct factor IC, not qrun/model IC.
        try:
            new_factors = self.process_factor_data(exp, quality_scope=quality_scope)
        except FactorEmptyError as e:
            logger.warning(
                "Factor implementation produced no usable factor matrix; "
                f"falling back to direct expression evaluation for archive/qrun input: {e}"
            )
            new_factors = self.build_direct_expression_quality_frame(
                exp,
                quality_scope,
                scope_kind="data" if QLIB_FACTOR_RUNNER_SETTINGS.archive_backtest_streaming else "quality",
            )
        if new_factors.empty:
            raise FactorEmptyError("No valid factor data found to merge.")
        if self.is_elitealpha_experiment(exp):
            new_factors = self.preselect_elitealpha_candidate_for_full_eval(
                exp,
                new_factors,
                quality_scope=quality_scope,
                qlib_config_name=config_name,
                template_folder_path=exp.experiment_workspace.template_folder_path,
            )
            if new_factors.empty:
                raise FactorEmptyError("No valid factor data remains after EliteAlpha preselection.")
        self.assign_factor_level_results(
            exp,
            new_factors,
            qlib_config_name=config_name,
            template_folder_path=exp.experiment_workspace.template_folder_path,
        )

        # AlphaAgent evaluates the new factors in the context of historical
        # SOTA factors. EliteAlpha scores candidates independently for archive
        # selection, so it deliberately skips this merge.
        if should_merge_sota and SOTA_factor is not None and not SOTA_factor.empty:
            new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
            if new_factors.empty:
                raise FactorEmptyError("No valid factor data found to merge.")
            combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
        else:
            combined_factors = new_factors

        if QLIB_FACTOR_RUNNER_SETTINGS.archive_backtest_streaming:
            combined_factors = self.prepare_archive_backtest_factor_matrix(combined_factors, quality_scope, scope_kind="data")
            if combined_factors.empty:
                raise FactorEmptyError("No valid factor data remains after archive backtest scoped filtering.")
            
        if len(combined_factors.columns) >= 2:
            pd.set_option('display.width', 1000)
            logger.info(f"Factor correlation: \n\n{combined_factors.corr()}\n")

        # Sort and nest the combined factors under 'feature'
        combined_factors = combined_factors.sort_index()
        combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
        self.log_factor_ic_input_debug(combined_factors)
        new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
        combined_factors.columns = new_columns

        logger.info(f"Factor values this round: \n\n{combined_factors.tail()}\n\n")

        # Save the combined factors to the workspace
        with open(exp.experiment_workspace.workspace_path / "combined_factors_df.pkl", "wb") as f:
            pickle.dump(combined_factors, f)


        # 执行回测，支持本地或Docker环境
        logger.info(f"Execute factor backtest (Use {'Local' if use_local else 'Docker container'}): {config_name}")
        
        result = exp.experiment_workspace.execute(
            qlib_config_name=config_name,
            use_local=use_local
        )
        
        logger.info(f"Backtesting results: \n{result.iloc[2:] if result is not None else 'None'}")
        exp.result = result

        return exp

    def should_merge_sota_factors(self, exp: QlibFactorExperiment) -> bool:
        """Return whether qrun should include historical SOTA factor columns."""
        return not self.is_elitealpha_experiment(exp)

    def preselect_elitealpha_candidate_for_full_eval(
        self,
        exp: QlibFactorExperiment,
        new_factors: pd.DataFrame,
        quality_scope: dict[str, Any],
        qlib_config_name: str,
        template_folder_path: Path | None = None,
    ) -> pd.DataFrame:
        """Use a one-train-year quick score to keep one EliteAlpha candidate."""
        task_names = [getattr(task, "factor_name", None) for task in getattr(exp, "sub_tasks", []) or []]
        task_names = [name for name in task_names if name]
        if len(task_names) <= 1:
            return new_factors

        quick_scope = self.archive_corr_evidence_scope(quality_scope)
        quick_factors = self.build_factor_quality_frame(exp, new_factors)
        if quick_factors.empty:
            logger.warning("EliteAlpha preselection skipped: cannot map factor tasks to generated factor columns.")
            return new_factors

        quick_factors = self.filter_factor_quality_frame(
            quick_factors,
            qlib_config_name=qlib_config_name,
            template_folder_path=template_folder_path,
            quality_scope=quick_scope,
        )
        if quick_factors.empty:
            logger.warning("EliteAlpha preselection skipped: no rows in one-year train evidence scope.")
            return new_factors

        quick_quality = self.calculate_factor_level_ic(
            quick_factors,
            quality_scope=quick_scope,
            allow_sign_flip=True,
        )
        if not quick_quality:
            logger.warning("EliteAlpha preselection skipped: one-year Rank IC produced no usable scores.")
            return new_factors

        archive = getattr(self, "current_elite_archive", None)
        rows: list[dict[str, Any]] = []
        registered_tasks = []
        task_by_name = {getattr(task, "factor_name", None): task for task in getattr(exp, "sub_tasks", []) or []}
        try:
            from alphaagent.scenarios.qlib.archive import (
                EliteRecord,
                evict_archive_factor_values,
                get_task_descriptor,
                register_archive_factor_values,
            )

            for factor_name, metrics in quick_quality.items():
                task = task_by_name.get(factor_name)
                if task is None or factor_name not in quick_factors.columns:
                    continue
                quality = metrics.get("Rank IC")
                if quality is None or not np.isfinite(quality):
                    continue

                evidence_values = pd.to_numeric(quick_factors.loc[:, factor_name], errors="coerce")
                if bool(metrics.get("sign_flipped")):
                    evidence_values = -evidence_values
                evidence_values = (
                    evidence_values.replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .sort_index()
                    .astype(np.float32, copy=True)
                )
                if evidence_values.empty:
                    continue

                register_archive_factor_values(task, evidence_values)
                registered_tasks.append(task)

                regularized_quality = float(quality)
                corr_penalty = 0.0
                max_corr = None
                top_k_avg_corr = None
                corr_top_k = None
                corr_match = None
                descriptor = get_task_descriptor(archive, task) if archive is not None else None
                if archive is not None and descriptor is not None:
                    record = EliteRecord.from_task(task, descriptor=descriptor, quality=float(quality))
                    regularized_quality, corr_info = archive.regularized_quality(
                        record,
                        exclude_descriptor=descriptor,
                    )
                    corr_penalty = float(corr_info.penalty)
                    max_corr = corr_info.max_abs_corr
                    top_k_avg_corr = corr_info.top_k_avg_abs_corr
                    corr_top_k = corr_info.corr_top_k
                    corr_match = corr_info.matched_factor_name

                rows.append(
                    {
                        "factor_name": factor_name,
                        "one_year_Rank IC": float(quality),
                        "regularized_quality": float(regularized_quality),
                        "corr_penalty": corr_penalty,
                        "top_k_avg_cross_category_corr": top_k_avg_corr,
                        "corr_top_k": corr_top_k,
                        "max_cross_category_corr": max_corr,
                        "corr_match": corr_match,
                        "sign_flipped_for_quick_score": bool(metrics.get("sign_flipped")),
                    }
                )
        except Exception as e:
            logger.warning(f"EliteAlpha one-year preselection failed; keeping all candidates: {e}")
            return new_factors

        if not rows:
            logger.warning("EliteAlpha preselection skipped: no candidate had a usable one-year regularized score.")
            return new_factors

        ranking = pd.DataFrame(rows).sort_values("regularized_quality", ascending=False)
        winner = str(ranking.iloc[0]["factor_name"])
        logger.info(
            "EliteAlpha one-year preselection ranking "
            f"(scope={quick_scope.get('start_time')} to {quick_scope.get('end_time')}):\n"
            f"{ranking.to_string(index=False)}"
        )
        logger.info(f"EliteAlpha preselection winner for full train/qrun/archive: {winner}")

        from alphaagent.scenarios.qlib.archive import evict_archive_factor_values

        for task in registered_tasks:
            if getattr(task, "factor_name", None) != winner:
                evict_archive_factor_values(task)

        self.keep_only_factor_candidate(exp, winner)
        return self.select_factor_columns(new_factors, exp, [winner])

    def archive_corr_evidence_scope(self, quality_scope: dict[str, Any]) -> dict[str, Any]:
        """Return the latest one-year slice inside the archive train segment."""
        scope = dict(quality_scope)
        start_time = scope.get("start_time")
        end_time = scope.get("end_time")
        if end_time is None:
            return scope
        end_time = pd.Timestamp(end_time)
        evidence_start = end_time - pd.DateOffset(years=1) + pd.Timedelta(days=1)
        if start_time is not None:
            evidence_start = max(evidence_start, pd.Timestamp(start_time))
        scope["start_time"] = evidence_start
        scope["end_time"] = end_time
        scope["segment_key"] = f"{scope.get('segment_key', 'train')}_one_year_preselect"
        return scope

    def keep_only_factor_candidate(self, exp: QlibFactorExperiment, factor_name: str) -> None:
        keep_indices = [
            idx
            for idx, task in enumerate(getattr(exp, "sub_tasks", []) or [])
            if getattr(task, "factor_name", None) == factor_name
        ]
        if not keep_indices:
            logger.warning(f"EliteAlpha preselection could not find task for winner {factor_name}.")
            return
        exp.sub_tasks = [exp.sub_tasks[idx] for idx in keep_indices]

        workspaces = list(getattr(exp, "sub_workspace_list", []) or [])
        if not workspaces:
            return
        selected_workspaces = []
        for idx, workspace in enumerate(workspaces):
            task = getattr(workspace, "target_task", None)
            workspace_factor_name = getattr(task, "factor_name", None) or getattr(task, "name", None)
            if workspace_factor_name == factor_name or idx in keep_indices:
                selected_workspaces.append(workspace)
        exp.sub_workspace_list = selected_workspaces

    def select_factor_columns(
        self,
        factors: pd.DataFrame,
        exp: QlibFactorExperiment,
        factor_names: list[str],
    ) -> pd.DataFrame:
        if factors is None or factors.empty:
            return factors
        selected = self.build_factor_quality_frame(exp, factors)
        matched = [name for name in factor_names if name in selected.columns]
        if matched:
            return selected.loc[:, matched]

        flat_columns = self.flatten_columns(factors.columns)
        original_columns = [column for column, flat in zip(factors.columns, flat_columns) if flat in factor_names]
        if original_columns:
            return factors.loc[:, original_columns]
        logger.warning(f"Could not select factor columns {factor_names}; keeping original factor matrix.")
        return factors

    @staticmethod
    def is_elitealpha_experiment(exp: QlibFactorExperiment) -> bool:
        if getattr(exp, "is_elitealpha", False) or getattr(exp, "skip_sota_factor_merge", False):
            return True

        for task in getattr(exp, "sub_tasks", []) or []:
            if (
                hasattr(task, "elite_descriptor")
                or hasattr(task, "elite_generation_mode")
                or hasattr(task, "elite_depth_bin")
                or hasattr(task, "elite_complexity_bin")
            ):
                return True
        return False

    def log_factor_ic_input_debug(self, factors: pd.DataFrame) -> None:
        """
        Log compact diagnostics for the factor matrix before Qlib trains a model.

        IC becomes NaN when the prediction or label has too few valid samples or
        zero cross-sectional variance on most dates. This checks the factor side
        before qrun so we can separate factor degeneration from Qlib/model issues.
        """
        if factors is None or factors.empty:
            logger.warning("IC debug: factor matrix is empty before qrun.")
            return

        debug_df = factors.copy()
        if isinstance(debug_df.columns, pd.MultiIndex):
            debug_df.columns = [".".join(str(part) for part in col) for col in debug_df.columns]

        numeric_df = debug_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        total_rows = len(numeric_df)
        rows = []
        has_datetime = "datetime" in numeric_df.index.names

        for col in numeric_df.columns:
            series = numeric_df[col]
            finite = series.dropna()
            row = {
                "factor": col,
                "rows": total_rows,
                "finite_rows": int(finite.shape[0]),
                "nan_or_inf_ratio": float(1 - finite.shape[0] / total_rows) if total_rows else 1.0,
                "unique_values": int(finite.nunique(dropna=True)),
                "mean": float(finite.mean()) if not finite.empty else np.nan,
                "std": float(finite.std()) if not finite.empty else np.nan,
                "min": float(finite.min()) if not finite.empty else np.nan,
                "max": float(finite.max()) if not finite.empty else np.nan,
            }

            if has_datetime:
                daily = series.groupby(level="datetime").agg(
                    valid_count=lambda s: int(s.notna().sum()),
                    unique_count=lambda s: int(s.dropna().nunique()),
                    std=lambda s: s.dropna().std(),
                )
                bad_daily = daily[
                    (daily["valid_count"] < 2)
                    | (daily["unique_count"] < 2)
                    | daily["std"].isna()
                    | (daily["std"] == 0)
                ]
                row.update(
                    {
                        "dates": int(daily.shape[0]),
                        "ic_bad_dates": int(bad_daily.shape[0]),
                        "ic_bad_date_ratio": float(bad_daily.shape[0] / daily.shape[0]) if len(daily) else 1.0,
                        "bad_date_examples": ", ".join(str(idx) for idx in bad_daily.head(5).index),
                    }
                )
            rows.append(row)

        summary = pd.DataFrame(rows)
        logger.info(f"IC input debug - factor matrix shape={factors.shape}, index_names={factors.index.names}")
        logger.info(f"IC input debug - per-factor summary:\n{summary.to_string(index=False)}")

    def assign_factor_level_results(
        self,
        exp: QlibFactorExperiment,
        new_factors: pd.DataFrame,
        qlib_config_name: str = "conf_cn_combined_kdd_ver.yaml",
        template_folder_path: Path | None = None,
    ) -> None:
        """
        Store per-factor quality in exp.sub_results.

        Qlib's normal result is an experiment-level score for the whole feature set.
        MAP-Elites needs a score for each candidate factor, so this follows the
        direct-factor IC convention used by EliteAlpha:
        train window + config market + label Rank-IC for archive replacement,
        plus validation Rank-IC for later manual prioritization.
        """
        try:
            quality_scope = self.load_quality_scope(
                qlib_config_name,
                template_folder_path,
                segment_key="train",
            )
            validation_scope = self.load_quality_scope(
                qlib_config_name,
                template_folder_path,
                segment_key="valid",
            )
            quality_factors = self.build_direct_expression_quality_frame(exp, quality_scope)
            if quality_factors.empty:
                quality_factors = self.build_factor_quality_frame(exp, new_factors)
                if quality_factors.empty:
                    logger.warning("Factor-level IC skipped: cannot map factor tasks to factor columns.")
                    return

            logger.info(
                "Factor-level IC scope from "
                f"{quality_scope.get('config_name')}: market={quality_scope.get('market')}, "
                f"segment={quality_scope.get('segment_key')}, "
                f"start={quality_scope.get('start_time')}, end={quality_scope.get('end_time')}, "
                f"label={quality_scope.get('label_expr')}"
            )
            quality_factors = self.filter_factor_quality_frame(
                quality_factors,
                qlib_config_name=qlib_config_name,
                template_folder_path=template_folder_path,
                quality_scope=quality_scope,
            )
            if quality_factors.empty:
                logger.warning(f"Factor-level IC skipped: no factor rows inside {qlib_config_name} quality scope.")
                return

            quality_by_factor = self.calculate_factor_level_ic(
                quality_factors,
                quality_scope=quality_scope,
                allow_sign_flip=True,
            )
            sub_results = {}
            for factor_name in quality_factors.columns:
                quality = quality_by_factor.get(factor_name)
                if quality is None:
                    logger.warning(f"Factor-level IC missing for {factor_name}.")
                    continue
                if quality.get("sign_flipped"):
                    self.apply_factor_sign_flip(exp, factor_name, new_factors)
                    quality_factors.loc[:, factor_name] = -pd.to_numeric(
                        quality_factors.loc[:, factor_name],
                        errors="coerce",
                    )
                else:
                    self.refresh_factor_task_complexity(exp, factor_name)
                self.register_archive_factor_values(
                    exp,
                    factor_name,
                    quality_factors.loc[:, factor_name],
                    quality_scope=quality_scope,
                )
                quality = self.with_segment_metrics(quality, "train")
                quality["quality_segment"] = "train"
                sub_results[factor_name] = quality

            validation_by_factor = self.calculate_validation_factor_results(
                exp,
                new_factors,
                validation_scope=validation_scope,
                qlib_config_name=qlib_config_name,
                template_folder_path=template_folder_path,
            )
            for factor_name, validation_quality in validation_by_factor.items():
                if factor_name not in sub_results:
                    continue
                validation_metrics = self.with_segment_metrics(validation_quality, "validation")
                sub_results[factor_name].update(
                    {
                        key: value
                        for key, value in validation_metrics.items()
                        if key.startswith("validation_")
                    }
                )

            for factor_name, metrics in sub_results.items():
                self.attach_factor_selection_metrics(exp, factor_name, metrics)

            exp.sub_results.update(sub_results)
            if sub_results:
                logger.info(f"Factor-level archive quality:\n{pd.DataFrame(sub_results).T.to_string()}")
            else:
                logger.warning("Factor-level IC produced no usable per-factor quality.")
        except Exception as e:
            logger.warning(f"Failed to calculate factor-level archive quality: {e}")

    def calculate_validation_factor_results(
        self,
        exp: QlibFactorExperiment,
        new_factors: pd.DataFrame,
        validation_scope: dict[str, Any],
        qlib_config_name: str,
        template_folder_path: Path | None,
    ) -> dict[str, dict[str, float]]:
        try:
            validation_factors = self.build_direct_expression_quality_frame(exp, validation_scope)
            if validation_factors.empty:
                validation_factors = self.build_factor_quality_frame(exp, new_factors)
                if validation_factors.empty:
                    logger.warning("Validation IC skipped: cannot map factor tasks to factor columns.")
                    return {}

            logger.info(
                "Factor-level validation IC scope from "
                f"{validation_scope.get('config_name')}: market={validation_scope.get('market')}, "
                f"segment={validation_scope.get('segment_key')}, "
                f"start={validation_scope.get('start_time')}, end={validation_scope.get('end_time')}, "
                f"label={validation_scope.get('label_expr')}"
            )
            validation_factors = self.filter_factor_quality_frame(
                validation_factors,
                qlib_config_name=qlib_config_name,
                template_folder_path=template_folder_path,
                quality_scope=validation_scope,
            )
            if validation_factors.empty:
                logger.warning(f"Validation IC skipped: no factor rows inside {qlib_config_name} validation scope.")
                return {}
            return self.calculate_factor_level_ic(
                validation_factors,
                quality_scope=validation_scope,
                allow_sign_flip=False,
            )
        except Exception as e:
            logger.warning(f"Failed to calculate validation factor quality: {e}")
            return {}

    @staticmethod
    def with_segment_metrics(metrics: dict[str, Any], segment_name: str) -> dict[str, Any]:
        enriched = dict(metrics)
        for key, value in metrics.items():
            enriched[f"{segment_name}_{key}"] = value
        return enriched

    def attach_factor_selection_metrics(
        self,
        exp: QlibFactorExperiment,
        factor_name: str,
        metrics: dict[str, Any],
    ) -> None:
        for task in getattr(exp, "sub_tasks", []) or []:
            if getattr(task, "factor_name", None) != factor_name:
                continue
            task.archive_quality_segment = metrics.get("quality_segment", "train")
            task.train_rank_ic = metrics.get("train_Rank IC", metrics.get("Rank IC"))
            task.train_rank_icir = metrics.get("train_Rank ICIR", metrics.get("Rank ICIR"))
            task.validation_rank_ic = metrics.get("validation_Rank IC")
            task.validation_rank_icir = metrics.get("validation_Rank ICIR")
            task.validation_raw_rank_ic = metrics.get("validation_raw_Rank IC")
            return

    def register_archive_factor_values(
        self,
        exp: QlibFactorExperiment,
        factor_name: str,
        values: pd.Series,
        quality_scope: dict[str, Any] | None = None,
    ) -> None:
        """Persist one train-year of factor values for archive corr checks."""
        series = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            logger.warning(f"Archive corr cache skipped for {factor_name}: no finite factor values.")
            return
        if isinstance(series.index, pd.MultiIndex):
            series = self.normalize_factor_index(series.to_frame(name=factor_name)).iloc[:, 0]
        series = self.select_archive_corr_evidence_year(series, quality_scope).sort_index().astype(np.float32, copy=True)
        if series.empty:
            logger.warning(f"Archive corr cache skipped for {factor_name}: no finite values in corr evidence year.")
            return

        for task in getattr(exp, "sub_tasks", []) or []:
            if getattr(task, "factor_name", None) != factor_name:
                continue
            from alphaagent.scenarios.qlib.archive import register_archive_factor_values

            values_path = self.write_archive_corr_values(task, factor_name, series)
            register_archive_factor_values(task, series, values_path=values_path)
            logger.info(
                f"Archive corr cache registered for {factor_name}: "
                f"rows={len(series)}, dtype={series.dtype}, path={values_path}"
            )
            return
        logger.warning(f"Archive corr cache skipped for {factor_name}: task not found.")

    def select_archive_corr_evidence_year(
        self,
        series: pd.Series,
        quality_scope: dict[str, Any] | None,
    ) -> pd.Series:
        """Use the latest one-year slice inside the train segment as corr evidence."""
        if series.empty or not isinstance(series.index, pd.MultiIndex):
            return series

        datetime_level = self.get_index_level_name(series.index, ("datetime",))
        if datetime_level is None:
            return series

        dates = pd.to_datetime(series.index.get_level_values(datetime_level))
        available_end = dates.max()
        if pd.isna(available_end):
            return series

        scope_end = None if quality_scope is None else quality_scope.get("end_time")
        scope_start = None if quality_scope is None else quality_scope.get("start_time")
        end_time = min(pd.Timestamp(available_end), pd.Timestamp(scope_end)) if scope_end is not None else pd.Timestamp(available_end)
        start_time = end_time - pd.DateOffset(years=1) + pd.Timedelta(days=1)
        if scope_start is not None:
            start_time = max(start_time, pd.Timestamp(scope_start))

        mask = (dates >= start_time) & (dates <= end_time)
        selected = series.loc[mask]
        logger.info(
            "Archive corr evidence year selected: "
            f"start={start_time.date()}, end={end_time.date()}, rows={len(selected)}/{len(series)}"
        )
        return selected

    def write_archive_corr_values(self, task, factor_name: str, series: pd.Series) -> Path:
        cache_dir = self.archive_corr_values_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        expression = str(getattr(task, "factor_expression", ""))
        digest = hashlib.md5(f"{factor_name}\0{expression}".encode("utf-8")).hexdigest()[:12]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(factor_name)).strip("._") or "factor"
        path = cache_dir / f"{safe_name}_{digest}.pkl"
        series.to_pickle(path)
        return path

    @staticmethod
    def archive_corr_values_dir() -> Path:
        root = os.getenv("LOG_TRACE_PATH") or os.getenv("log_trace_path") or os.getenv("ALPHAAGENT_LOG_TRACE_PATH")
        if root:
            return Path(root).expanduser() / "archive_corr_values"
        return Path.cwd() / "log" / "archive_corr_values"

    def apply_factor_sign_flip(
        self,
        exp: QlibFactorExperiment,
        factor_name: str,
        factor_values: pd.DataFrame | None = None,
    ) -> None:
        """Keep archived expressions aligned with sign-flipped direct IC quality."""
        for task in getattr(exp, "sub_tasks", []) or []:
            if getattr(task, "factor_name", None) != factor_name:
                continue
            expression = getattr(task, "factor_expression", None)
            if not expression:
                return

            task.factor_expression = self.flip_leading_sign(str(expression))
            self.refresh_task_complexity(task)
            self.flip_factor_value_column(factor_values, exp, factor_name)
            description = getattr(task, "factor_description", "")
            if description and "Sign-adjusted" not in description:
                task.factor_description = f"Sign-adjusted for positive Rank IC. {description}"
            logger.info(
                f"Sign-adjusted factor for {factor_name} because direct Rank IC was negative: "
                f"expression={task.factor_expression}, "
                f"ast_depth={getattr(task, 'factor_ast_depth', None)}, "
                f"complexity_value={getattr(task, 'factor_complexity_value', None)}"
            )
            return

    def refresh_factor_task_complexity(self, exp: QlibFactorExperiment, factor_name: str) -> None:
        for task in getattr(exp, "sub_tasks", []) or []:
            if getattr(task, "factor_name", None) == factor_name:
                self.refresh_task_complexity(task)
                return

    @staticmethod
    def flip_leading_sign(expression: str) -> str:
        expression = expression.strip()
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
        return f"-1 * ({expression})"

    @classmethod
    def expression_without_leading_sign(cls, expression: str) -> str:
        expression = expression.strip()
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

    def refresh_task_complexity(self, task) -> None:
        expression = getattr(task, "factor_expression", None)
        if not expression:
            return
        expression_for_depth = self.expression_without_leading_sign(str(expression))
        try:
            from alphaagent.components.coder.factor_coder.factor_ast import count_all_nodes, count_depth

            task.factor_ast_depth = count_depth(expression_for_depth)
            task.factor_ast_node_count = count_all_nodes(expression_for_depth)
            metric = getattr(task, "factor_complexity_metric", None)
            if metric == "vertex":
                task.factor_complexity_value = task.factor_ast_node_count
            else:
                task.factor_complexity_value = task.factor_ast_depth
                task.factor_complexity_metric = metric or "depth"
        except Exception as e:
            logger.warning(f"Failed to refresh sign-adjusted complexity for {getattr(task, 'factor_name', None)}: {e}")

    def flip_factor_value_column(
        self,
        factors: pd.DataFrame | None,
        exp: QlibFactorExperiment,
        factor_name: str,
    ) -> None:
        if factors is None or factors.empty:
            return
        flat_columns = self.flatten_columns(factors.columns)
        for original_column, flat_column in zip(list(factors.columns), flat_columns):
            if flat_column == factor_name:
                factors.loc[:, original_column] = -pd.to_numeric(factors.loc[:, original_column], errors="coerce")
                return

        task_names = [getattr(task, "factor_name", None) for task in getattr(exp, "sub_tasks", []) or []]
        if len(task_names) == len(flat_columns) and factor_name in task_names:
            original_column = list(factors.columns)[task_names.index(factor_name)]
            factors.loc[:, original_column] = -pd.to_numeric(factors.loc[:, original_column], errors="coerce")
            return

        logger.warning(f"Could not find factor value column to sign-adjust for {factor_name}.")

    def build_factor_quality_frame(self, exp: QlibFactorExperiment, factors: pd.DataFrame) -> pd.DataFrame:
        quality_factors = factors.copy()
        flat_columns = self.flatten_columns(quality_factors.columns)
        quality_factors.columns = flat_columns
        task_names = [task.factor_name for task in exp.sub_tasks]
        if len(flat_columns) == len(task_names):
            quality_factors.columns = task_names
            return quality_factors

        matched = [task_name for task_name in task_names if task_name in quality_factors.columns]
        return quality_factors.loc[:, matched]

    def build_direct_expression_quality_frame(
        self,
        exp: QlibFactorExperiment,
        quality_scope: dict[str, Any],
        scope_kind: str = "quality",
    ) -> pd.DataFrame:
        expressions = self.collect_factor_expressions(exp)
        if not expressions:
            logger.warning(
                "Factor-level IC direct eval skipped: no factor expressions found "
                f"for tasks={[getattr(task, 'factor_name', None) for task in exp.sub_tasks]} "
                f"workspaces={len(getattr(exp, 'sub_workspace_list', []) or [])}"
            )
            return pd.DataFrame()

        daily_pv_path = DIRNAME.parent / "experiment" / "factor_data_template" / "daily_pv_all.h5"
        if not daily_pv_path.exists():
            logger.warning(f"Factor-level IC direct eval skipped: missing {daily_pv_path}")
            return pd.DataFrame()

        logger.info(f"Factor-level IC direct eval source: {daily_pv_path}")
        try:
            daily_pv = pd.read_hdf(daily_pv_path, key="data")
        except Exception as e:
            logger.warning(f"Factor-level IC direct eval skipped: failed to read {daily_pv_path}: {e}")
            return pd.DataFrame()

        if QLIB_FACTOR_RUNNER_SETTINGS.archive_backtest_streaming:
            return self.build_direct_expression_frame_streaming(expressions, daily_pv, quality_scope, scope_kind)

        series_by_name = {}
        uppercase_instruments = bool(quality_scope.get("uppercase_instruments", False))
        for factor_name, expression in expressions.items():
            try:
                series_by_name[factor_name] = self.evaluate_factor_expression_direct(
                    str(expression),
                    daily_pv,
                    upper_instrument=uppercase_instruments,
                )
            except Exception as e:
                logger.warning(f"Factor-level IC direct eval failed for {factor_name}: {e}")

        if not series_by_name:
            return pd.DataFrame()
        return pd.concat(series_by_name, axis=1)

    def build_direct_expression_frame_streaming(
        self,
        expressions: dict[str, str],
        daily_pv: pd.DataFrame,
        quality_scope: dict[str, Any],
        scope_kind: str,
    ) -> pd.DataFrame:
        target_start, target_end = self.get_scope_date_range(quality_scope, scope_kind)
        source = self.prepare_direct_eval_source(daily_pv, quality_scope, target_start, target_end)
        frames = []
        uppercase_instruments = bool(quality_scope.get("uppercase_instruments", False))
        for factor_name, expression in expressions.items():
            try:
                series = self.evaluate_factor_expression_direct(
                    str(expression),
                    source,
                    upper_instrument=uppercase_instruments,
                )
                frame = series.rename(factor_name).to_frame()
                frame = self.filter_factor_frame_to_scope(
                    frame,
                    quality_scope,
                    target_start=target_start,
                    target_end=target_end,
                )
                if not frame.empty:
                    frames.append(frame.astype(np.float32))
                del series, frame
            except Exception as e:
                logger.warning(f"Factor-level IC direct eval failed for {factor_name}: {e}")

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)

    def collect_factor_expressions(self, exp: QlibFactorExperiment) -> dict[str, str]:
        expressions = {
            task.factor_name: str(getattr(task, "factor_expression"))
            for task in exp.sub_tasks
            if getattr(task, "factor_expression", None)
        }
        for workspace in getattr(exp, "sub_workspace_list", []) or []:
            task = getattr(workspace, "target_task", None)
            factor_name = getattr(task, "factor_name", None) or getattr(task, "name", None)
            if not factor_name or factor_name in expressions:
                continue
            expression = getattr(task, "factor_expression", None)
            if expression:
                expressions[str(factor_name)] = str(expression)
                continue

            factor_code = None
            code_dict = getattr(workspace, "code_dict", None) or {}
            if "factor.py" in code_dict:
                factor_code = str(code_dict["factor.py"])
            else:
                factor_path = getattr(workspace, "workspace_path", None)
                if factor_path is not None:
                    factor_file = Path(factor_path) / "factor.py"
                    try:
                        factor_code = factor_file.read_text(encoding="utf-8")
                    except OSError:
                        factor_code = None
            expression = self.extract_expression_from_factor_code(factor_code)
            if expression:
                expressions[str(factor_name)] = expression
        return expressions

    @staticmethod
    def extract_expression_from_factor_code(factor_code: str | None) -> str | None:
        if not factor_code:
            return None
        match = re.search(r"^\s*expr\s*=\s*(['\"])(?P<expr>.+?)\1", factor_code, flags=re.MULTILINE)
        if match is None:
            return None
        return match.group("expr")

    def evaluate_factor_expression_direct(
        self,
        expr_str: str,
        daily_pv: pd.DataFrame,
        upper_instrument: bool = True,
    ) -> pd.Series:
        from alphaagent.components.coder.factor_coder import function_lib
        from alphaagent.components.coder.factor_coder.expr_parser import parse_expression, parse_symbol

        expr = parse_symbol(expr_str, daily_pv.columns)
        expr = parse_expression(expr)
        for col in daily_pv.columns:
            expr = expr.replace(col[1:], f"daily_pv[{col!r}]")

        namespace = {
            name: getattr(function_lib, name)
            for name in dir(function_lib)
            if not name.startswith("__")
        }
        namespace.update({"np": np, "pd": pd, "daily_pv": daily_pv})
        result = eval(expr, namespace)
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        if not isinstance(result, pd.Series):
            result = pd.Series(result, index=daily_pv.index)

        result = pd.to_numeric(result, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        result.index = result.index.set_names(["datetime", "instrument"])
        if upper_instrument:
            result.index = pd.MultiIndex.from_arrays(
                [
                    result.index.get_level_values("datetime"),
                    result.index.get_level_values("instrument").map(str).str.upper(),
                ],
                names=["datetime", "instrument"],
            )
        dtype = np.float32 if QLIB_FACTOR_RUNNER_SETTINGS.archive_backtest_streaming else np.float64
        return result.astype(dtype)

    def prepare_direct_eval_source(
        self,
        daily_pv: pd.DataFrame,
        quality_scope: dict[str, Any],
        target_start: pd.Timestamp | None,
        target_end: pd.Timestamp | None,
    ) -> pd.DataFrame:
        start_time = self.apply_warmup_start(target_start)
        source = self.filter_factor_frame_to_scope(
            daily_pv,
            quality_scope,
            target_start=start_time,
            target_end=target_end,
        )
        logger.info(
            "archive backtest direct eval source scoped with warmup: "
            f"rows {len(source)}/{len(daily_pv)}, "
            f"start={start_time}, end={target_end}, market={quality_scope.get('market')}"
        )
        return source

    def apply_warmup_start(self, start_time: pd.Timestamp | None) -> pd.Timestamp | None:
        if start_time is None:
            return None
        warmup_days = max(int(QLIB_FACTOR_RUNNER_SETTINGS.archive_backtest_warmup_days), 0)
        return pd.Timestamp(start_time) - pd.Timedelta(days=warmup_days)

    def get_scope_date_range(
        self,
        quality_scope: dict[str, Any],
        scope_kind: str = "quality",
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if scope_kind == "data":
            return (
                quality_scope.get("data_start_time") or quality_scope.get("start_time"),
                quality_scope.get("data_end_time") or quality_scope.get("end_time"),
            )
        return quality_scope.get("start_time"), quality_scope.get("end_time")

    def prepare_archive_backtest_factor_matrix(
        self,
        factors: pd.DataFrame,
        quality_scope: dict[str, Any],
        scope_kind: str = "data",
    ) -> pd.DataFrame:
        if factors is None or factors.empty:
            return factors
        start_time, end_time = self.get_scope_date_range(quality_scope, scope_kind)
        filtered = self.filter_factor_frame_to_scope(
            factors,
            quality_scope,
            target_start=start_time,
            target_end=end_time,
        )
        if filtered.empty:
            return filtered
        return filtered.astype(np.float32)

    def filter_factor_frame_to_scope(
        self,
        factors: pd.DataFrame,
        quality_scope: dict[str, Any],
        target_start: pd.Timestamp | None,
        target_end: pd.Timestamp | None,
    ) -> pd.DataFrame:
        filtered = self.normalize_factor_index(factors)
        filtered = self.align_factor_instrument_case(filtered, quality_scope)

        datetime_level = self.get_index_level_name(filtered.index, ("datetime",))
        if datetime_level is not None and target_start is not None and target_end is not None:
            dates = pd.to_datetime(filtered.index.get_level_values(datetime_level))
            mask = (dates >= pd.Timestamp(target_start)) & (dates <= pd.Timestamp(target_end))
            filtered = filtered.loc[mask]

        scope = dict(quality_scope)
        scope["start_time"] = target_start
        scope["end_time"] = target_end
        return self.filter_factor_quality_universe(filtered, scope)

    def filter_factor_quality_frame(
        self,
        factors: pd.DataFrame,
        qlib_config_name: str,
        template_folder_path: Path | None = None,
        quality_scope: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        quality_scope = quality_scope or self.load_quality_scope(qlib_config_name, template_folder_path)
        filtered = self.normalize_factor_index(factors)
        filtered = self.align_factor_instrument_case(filtered, quality_scope)

        date_range = (quality_scope.get("start_time"), quality_scope.get("end_time"))
        if date_range[0] is None or date_range[1] is None:
            return self.filter_factor_quality_universe(filtered, quality_scope)

        datetime_level = self.get_index_level_name(filtered.index, ("datetime",))
        if datetime_level is None:
            logger.warning(
                f"Factor-level IC date filter skipped: factor index has no datetime level, got {filtered.index.names}"
            )
            return self.filter_factor_quality_universe(filtered, quality_scope)

        start_time, end_time = date_range
        dates = pd.to_datetime(filtered.index.get_level_values(datetime_level))
        mask = (dates >= start_time) & (dates <= end_time)
        date_filtered = filtered.loc[mask]
        logger.info(
            "Factor-level IC date range from "
            f"{qlib_config_name}: {start_time.date()} to {end_time.date()}, "
            f"rows {len(date_filtered)}/{len(filtered)}"
        )
        return self.filter_factor_quality_universe(date_filtered, quality_scope)

    def load_quality_scope(
        self,
        qlib_config_name: str,
        template_folder_path: Path | None = None,
        segment_key: str = "train",
    ) -> dict[str, Any]:
        config_path = (template_folder_path or DIRNAME.parent / "experiment" / "factor_template") / qlib_config_name
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Factor-level IC config fallback: failed to read {config_path}: {e}")
            config = {}

        start_time, end_time = self.extract_quality_date_range(config, segment_key=segment_key)
        data_start_time, data_end_time = self.extract_data_handler_date_range(config)
        qlib_init = config.get("qlib_init", {}) or {}
        market = (
            config.get("market")
            or (config.get("data_handler_config", {}) or {}).get("instruments")
            or "csi500"
        )
        region = qlib_init.get("region", "cn")
        provider_uri = os.environ.get("QLIB_PROVIDER_URI", qlib_init.get("provider_uri", "~/.qlib/qlib_data/cn_data"))
        label_expr = self.extract_label_expression(config) or "Ref($close, -2)/Ref($close, -1) - 1"
        uppercase_instruments = str(market).lower() not in {"all", "csi300_ext"}
        return {
            "config_path": config_path,
            "config_name": qlib_config_name,
            "segment_key": segment_key,
            "start_time": start_time,
            "end_time": end_time,
            "data_start_time": data_start_time,
            "data_end_time": data_end_time,
            "market": market,
            "region": region,
            "provider_uri": provider_uri,
            "label_expr": label_expr,
            "uppercase_instruments": uppercase_instruments,
        }

    def extract_quality_date_range(
        self,
        config: dict[str, Any],
        segment_key: str = "train",
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        segments = config.get("task", {}).get("dataset", {}).get("kwargs", {}).get("segments", {})
        segment = segments.get(segment_key)
        if isinstance(segment, (list, tuple)) and len(segment) >= 2:
            return pd.Timestamp(segment[0]), pd.Timestamp(segment[1])

        backtest = config.get("port_analysis_config", {}).get("backtest", {})
        if "start_time" in backtest and "end_time" in backtest:
            return pd.Timestamp(backtest["start_time"]), pd.Timestamp(backtest["end_time"])

        data_handler = config.get("data_handler_config", {})
        if "start_time" in data_handler and "end_time" in data_handler:
            return pd.Timestamp(data_handler["start_time"]), pd.Timestamp(data_handler["end_time"])

        return None, None

    def extract_label_expression(self, config: dict[str, Any]) -> str | None:
        data_loader = (config.get("data_handler_config", {}) or {}).get("data_loader", {}) or {}
        kwargs = data_loader.get("kwargs", {}) or {}
        loaders = kwargs.get("dataloader_l")
        if isinstance(loaders, list):
            for loader in loaders:
                label = ((loader or {}).get("kwargs", {}) or {}).get("config", {}).get("label")
                expr = self.first_label_expression(label)
                if expr:
                    return expr

        label = (kwargs.get("config", {}) or {}).get("label")
        return self.first_label_expression(label)

    @staticmethod
    def first_label_expression(label_config: Any) -> str | None:
        if isinstance(label_config, str):
            return label_config
        if isinstance(label_config, (list, tuple)) and label_config:
            first = label_config[0]
            if isinstance(first, str):
                return first
            if isinstance(first, (list, tuple)) and first and isinstance(first[0], str):
                return first[0]
        return None

    @staticmethod
    def extract_data_handler_date_range(config: dict[str, Any]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        data_handler = config.get("data_handler_config", {}) or {}
        if "start_time" in data_handler and "end_time" in data_handler:
            return pd.Timestamp(data_handler["start_time"]), pd.Timestamp(data_handler["end_time"])
        return None, None

    def filter_factor_quality_universe(self, factors: pd.DataFrame, quality_scope: dict[str, Any]) -> pd.DataFrame:
        market = quality_scope.get("market")
        if market is None or str(market).lower() == "all":
            return factors

        instrument_level = self.get_index_level_name(factors.index, ("instrument", "code", "symbol"))
        if instrument_level is None:
            logger.warning(
                f"Factor-level IC universe filter skipped: factor index has no instrument level, got {factors.index.names}"
            )
            return factors

        try:
            from qlib.data import D

            self.init_qlib_from_scope(quality_scope)
            start_time = quality_scope.get("start_time")
            end_time = quality_scope.get("end_time")
            instruments = D.list_instruments(
                D.instruments(str(market)),
                as_list=True,
                start_time=start_time.strftime("%Y-%m-%d") if start_time is not None else None,
                end_time=end_time.strftime("%Y-%m-%d") if end_time is not None else None,
            )
        except Exception as e:
            logger.warning(f"Factor-level IC universe filter skipped for market={market!r}: {e}")
            return factors

        if quality_scope.get("uppercase_instruments", False):
            universe = {str(instrument).upper() for instrument in instruments}
        else:
            universe = {str(instrument) for instrument in instruments}

        mask = factors.index.get_level_values(instrument_level).map(str).isin(universe)
        filtered = factors.loc[mask]
        logger.info(
            f"Factor-level IC universe from {quality_scope.get('config_name')}: market={market}, "
            f"rows {len(filtered)}/{len(factors)}, instruments={len(universe)}"
        )
        return filtered

    def align_factor_instrument_case(self, factors: pd.DataFrame, quality_scope: dict[str, Any]) -> pd.DataFrame:
        if not quality_scope.get("uppercase_instruments", False):
            return factors
        return self.transform_instrument_index(factors, lambda value: str(value).upper())

    def transform_instrument_index(self, df: pd.DataFrame, transform) -> pd.DataFrame:
        instrument_level = self.get_index_level_name(df.index, ("instrument", "code", "symbol"))
        if instrument_level is None or not isinstance(df.index, pd.MultiIndex):
            return df

        arrays = []
        for name in df.index.names:
            values = df.index.get_level_values(name)
            if name == instrument_level:
                values = values.map(transform)
            arrays.append(values)

        transformed = df.copy()
        transformed.index = pd.MultiIndex.from_arrays(arrays, names=df.index.names)
        return transformed.sort_index()

    def init_qlib_from_scope(self, quality_scope: dict[str, Any]) -> None:
        import qlib

        provider_uri = quality_scope.get("provider_uri") or "~/.qlib/qlib_data/cn_data"
        region = quality_scope.get("region") or "cn"
        qlib.init(provider_uri=str(Path(provider_uri).expanduser()), region=region)

    def load_quality_date_range(
        self,
        qlib_config_name: str,
        template_folder_path: Path | None = None,
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        scope = self.load_quality_scope(qlib_config_name, template_folder_path)
        start_time, end_time = scope.get("start_time"), scope.get("end_time")
        if start_time is not None and end_time is not None:
            return start_time, end_time
        logger.warning(f"Factor-level IC date filter skipped: no test/backtest date range found in {scope.get('config_path')}.")
        return None

    def calculate_factor_level_ic(
        self,
        factors: pd.DataFrame,
        quality_scope: dict[str, Any] | None = None,
        allow_sign_flip: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Compute direct-factor IC. Sign flip is allowed only on the archive selection segment."""
        factor_df = factors.copy()
        factor_df.columns = self.flatten_columns(factor_df.columns)
        factor_df = self.normalize_factor_index(factor_df)
        label = self.load_qlib_label(factor_df.index, quality_scope=quality_scope)
        if label is None or label.empty:
            raise ValueError("label series is empty")

        results = {}
        for column in factor_df.columns:
            pair = pd.concat(
                [
                    pd.to_numeric(factor_df[column], errors="coerce").rename("factor"),
                    label.rename("label"),
                ],
                axis=1,
                join="inner",
            ).replace([np.inf, -np.inf], np.nan).dropna()
            if pair.empty:
                results[column] = {"IC": np.nan, "Rank IC": np.nan}
                continue

            daily_rows = []
            for _, group in pair.groupby(level="datetime"):
                if len(group) < 5:
                    continue
                factor_std = group["factor"].std()
                label_std = group["label"].std()
                if pd.isna(factor_std) or pd.isna(label_std) or factor_std == 0 or label_std == 0:
                    continue
                daily_rows.append(
                    (
                        group["factor"].corr(group["label"]),
                        group["factor"].rank().corr(group["label"].rank()),
                    )
                )

            daily_ic = pd.DataFrame(daily_rows, columns=["IC", "Rank IC"])
            if daily_ic.empty:
                results[column] = {"IC": np.nan, "Rank IC": np.nan, "ICIR": np.nan, "Rank ICIR": np.nan}
                continue

            ic_mean = float(daily_ic["IC"].mean()) if daily_ic["IC"].notna().any() else np.nan
            rank_ic_mean = float(daily_ic["Rank IC"].mean()) if daily_ic["Rank IC"].notna().any() else np.nan
            raw_ic_mean = ic_mean
            raw_rank_ic_mean = rank_ic_mean
            sign_flipped = bool(allow_sign_flip and pd.notna(rank_ic_mean) and rank_ic_mean < 0)
            if sign_flipped:
                daily_ic["IC"] = -daily_ic["IC"]
                daily_ic["Rank IC"] = -daily_ic["Rank IC"]
                ic_mean = -ic_mean if pd.notna(ic_mean) else ic_mean
                rank_ic_mean = -rank_ic_mean

            ic_std = daily_ic["IC"].std()
            rank_ic_std = daily_ic["Rank IC"].std()
            results[column] = {
                "raw_IC": raw_ic_mean,
                "raw_Rank IC": raw_rank_ic_mean,
                "IC": ic_mean,
                "Rank IC": rank_ic_mean,
                "ICIR": float(ic_mean / (ic_std + 1e-12)) if pd.notna(ic_mean) and pd.notna(ic_std) else np.nan,
                "Rank ICIR": float(rank_ic_mean / (rank_ic_std + 1e-12))
                if pd.notna(rank_ic_mean) and pd.notna(rank_ic_std)
                else np.nan,
                "sign_flipped": sign_flipped,
            }
        return results

    def load_qlib_label(self, index: pd.Index, quality_scope: dict[str, Any] | None = None) -> pd.Series:
        datetime_level = self.get_index_level_name(index, ("datetime",))
        instrument_level = self.get_index_level_name(index, ("instrument", "code", "symbol"))
        if datetime_level is None or instrument_level is None:
            raise ValueError(f"factor index must include datetime and instrument levels, got {index.names}")

        dates = index.get_level_values(datetime_level)
        instruments = sorted({str(inst) for inst in index.get_level_values(instrument_level)})
        start_time = pd.Timestamp(dates.min()).strftime("%Y-%m-%d")
        end_time = pd.Timestamp(dates.max()).strftime("%Y-%m-%d")

        from qlib.data import D

        quality_scope = quality_scope or {}
        self.init_qlib_from_scope(quality_scope)
        label_expr = quality_scope.get("label_expr") or "Ref($close, -2)/Ref($close, -1) - 1"
        label_df = D.features(
            instruments,
            [label_expr],
            start_time=start_time,
            end_time=end_time,
            freq="day",
        )
        if label_df.empty:
            return pd.Series(dtype=float, name="label")

        label = pd.to_numeric(label_df.iloc[:, 0], errors="coerce").rename("label")
        label = self.normalize_factor_index(label.to_frame()).iloc[:, 0]
        return label

    @staticmethod
    def flatten_columns(columns) -> list[str]:
        if isinstance(columns, pd.MultiIndex):
            return [".".join(str(part) for part in col if str(part) != "") for col in columns]
        return [str(col) for col in columns]

    @staticmethod
    def get_index_level_name(index: pd.Index, candidates: tuple[str, ...]) -> str | None:
        names = list(index.names)
        for candidate in candidates:
            if candidate in names:
                return candidate
        return None

    def normalize_factor_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.MultiIndex):
            return df.sort_index()

        names = list(df.index.names)
        datetime_level = self.get_index_level_name(df.index, ("datetime",))
        instrument_level = self.get_index_level_name(df.index, ("instrument", "code", "symbol"))
        if datetime_level is None or instrument_level is None:
            return df.sort_index()

        normalized = df.copy()
        if instrument_level != "instrument":
            names[names.index(instrument_level)] = "instrument"
        normalized.index = normalized.index.set_names(names)
        return normalized.reorder_levels(["datetime", "instrument"]).sort_index()

    def process_factor_data(
        self,
        exp_or_list: List[QlibFactorExperiment] | QlibFactorExperiment,
        quality_scope: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Process and combine factor data from experiment implementations.

        Args:
            exp (ASpecificExp): The experiment containing factor data.

        Returns:
            pd.DataFrame: Combined factor data without NaN values.
        """
        if isinstance(exp_or_list, QlibFactorExperiment):
            exp_or_list = [exp_or_list]
        factor_dfs = []

        # Collect all exp's dataframes
        for exp in exp_or_list:
            # Iterate over sub-implementations and execute them to get each factor data
            message_and_df_list = multiprocessing_wrapper(
                [(implementation.execute, ("All",)) for implementation in exp.sub_workspace_list],
                n=RD_AGENT_SETTINGS.multi_proc_n,
            )
            for message, df in message_and_df_list:
                # Check if factor generation was successful
                if df is not None and "datetime" in df.index.names:
                    time_diff = df.index.get_level_values("datetime").to_series().diff().dropna().unique()
                    if pd.Timedelta(minutes=1) not in time_diff:
                        if QLIB_FACTOR_RUNNER_SETTINGS.archive_backtest_streaming and quality_scope is not None:
                            df = self.prepare_archive_backtest_factor_matrix(df, quality_scope, scope_kind="data")
                        if not df.empty:
                            factor_dfs.append(df)

        # Combine all successful factor data
        if factor_dfs:
            return pd.concat(factor_dfs, axis=1)
        else:
            raise FactorEmptyError("No valid factor data found to merge.")
