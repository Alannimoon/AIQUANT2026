import pickle
from pathlib import Path
from typing import Any, List
import os
import numpy as np
import pandas as pd
from pandarallel import pandarallel

from alphaagent.core.conf import RD_AGENT_SETTINGS
from alphaagent.core.utils import cache_with_pickle, multiprocessing_wrapper

pandarallel.initialize(verbose=1)

from alphaagent.components.runner import CachedRunner
from alphaagent.core.exception import FactorEmptyError
from alphaagent.log import logger
from alphaagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment

DIRNAME = Path(__file__).absolute().resolve().parent
DIRNAME_local = Path.cwd()
FACTOR_LEVEL_QUALITY_CACHE_VERSION = "paper_direct_ic_v1"


def get_factor_runner_cache_key(self, exp: QlibFactorExperiment, **kwargs) -> str:
    base_key = CachedRunner.get_cache_key(self, exp, **kwargs)
    return f"{FACTOR_LEVEL_QUALITY_CACHE_VERSION}_{base_key}"

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
        config_name = f"conf.yaml" if len(exp.based_experiments) == 0 else "conf_cn_combined_kdd_ver.yaml"
        if not hasattr(exp.experiment_workspace, "template_folder_path"):
            exp.experiment_workspace.template_folder_path = DIRNAME.parent / "experiment" / "factor_template"
        
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1], use_local=use_local)

        if exp.based_experiments:
            SOTA_factor = None
            if len(exp.based_experiments) > 1:
                SOTA_factor = self.process_factor_data(exp.based_experiments)

            # Process the new factors data
            new_factors = self.process_factor_data(exp)
            if new_factors.empty:
                raise FactorEmptyError("No valid factor data found to merge.")
            self.assign_factor_level_results(
                exp,
                new_factors,
                qlib_config_name=config_name,
                template_folder_path=exp.experiment_workspace.template_folder_path,
            )

            # Combine the SOTA factor and new factors if SOTA factor exists
            if False: # SOTA_factor is not None and not SOTA_factor.empty:
                new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
                if new_factors.empty:
                    raise FactorEmptyError("No valid factor data found to merge.")
                combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
            else:
                combined_factors = new_factors
                
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
        direct-factor IC convention used by scripts/eval_for_paper.py:
        config test window + config market + label IC + Rank-IC sign flip.
        """
        try:
            quality_factors = self.build_factor_quality_frame(exp, new_factors)
            if quality_factors.empty:
                logger.warning("Factor-level IC skipped: cannot map factor tasks to factor columns.")
                return

            quality_scope = self.load_quality_scope(qlib_config_name, template_folder_path)
            logger.info(
                "Factor-level IC scope from "
                f"{quality_scope.get('config_name')}: market={quality_scope.get('market')}, "
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

            quality_by_factor = self.calculate_factor_level_ic(quality_factors, quality_scope=quality_scope)
            sub_results = {}
            for factor_name in quality_factors.columns:
                quality = quality_by_factor.get(factor_name)
                if quality is None:
                    logger.warning(f"Factor-level IC missing for {factor_name}.")
                    continue
                sub_results[factor_name] = quality

            exp.sub_results.update(sub_results)
            if sub_results:
                logger.info(f"Factor-level archive quality:\n{pd.DataFrame(sub_results).T.to_string()}")
            else:
                logger.warning("Factor-level IC produced no usable per-factor quality.")
        except Exception as e:
            logger.warning(f"Failed to calculate factor-level archive quality: {e}")

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
    ) -> dict[str, Any]:
        config_path = (template_folder_path or DIRNAME.parent / "experiment" / "factor_template") / qlib_config_name
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Factor-level IC config fallback: failed to read {config_path}: {e}")
            config = {}

        start_time, end_time = self.extract_quality_date_range(config)
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
            "start_time": start_time,
            "end_time": end_time,
            "market": market,
            "region": region,
            "provider_uri": provider_uri,
            "label_expr": label_expr,
            "uppercase_instruments": uppercase_instruments,
        }

    def extract_quality_date_range(self, config: dict[str, Any]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        segments = config.get("task", {}).get("dataset", {}).get("kwargs", {}).get("segments", {})
        test_segment = segments.get("test")
        if isinstance(test_segment, (list, tuple)) and len(test_segment) >= 2:
            return pd.Timestamp(test_segment[0]), pd.Timestamp(test_segment[1])

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
    ) -> dict[str, dict[str, float]]:
        """Compute direct-factor IC with the same sign convention as eval_for_paper.py."""
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
            sign_flipped = bool(pd.notna(rank_ic_mean) and rank_ic_mean < 0)
            if sign_flipped:
                daily_ic["IC"] = -daily_ic["IC"]
                daily_ic["Rank IC"] = -daily_ic["Rank IC"]
                ic_mean = -ic_mean if pd.notna(ic_mean) else ic_mean
                rank_ic_mean = -rank_ic_mean

            ic_std = daily_ic["IC"].std()
            rank_ic_std = daily_ic["Rank IC"].std()
            results[column] = {
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

    def process_factor_data(self, exp_or_list: List[QlibFactorExperiment] | QlibFactorExperiment) -> pd.DataFrame:
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
                        factor_dfs.append(df)

        # Combine all successful factor data
        if factor_dfs:
            return pd.concat(factor_dfs, axis=1)
        else:
            raise FactorEmptyError("No valid factor data found to merge.")
