import pickle
from pathlib import Path
from typing import List
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

    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibFactorExperiment, use_local: bool = True) -> QlibFactorExperiment:
        
        """
        Generate the experiment by processing and combining factor data,
        then passing the combined data to Docker or local environment for backtest results.
        """
        
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
            self.assign_factor_level_results(exp, new_factors)

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
        config_name = f"conf.yaml" if len(exp.based_experiments) == 0 else "conf_cn_combined_kdd_ver.yaml"
        logger.info(f"Execute factor backtest (Use {'Local' if use_local else 'Docker container'}): {config_name}")
        if not hasattr(exp.experiment_workspace, "template_folder_path"):
            exp.experiment_workspace.template_folder_path = DIRNAME.parent / "experiment" / "factor_template"
        
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

    def assign_factor_level_results(self, exp: QlibFactorExperiment, new_factors: pd.DataFrame) -> None:
        """
        Store per-factor quality in exp.sub_results.

        Qlib's normal result is an experiment-level score for the whole feature
        set. MAP-Elites needs a score for each candidate factor, so we use the
        raw factor-label cross-sectional IC as a cheap factor-level proxy.
        """
        try:
            quality_factors = self.build_factor_quality_frame(exp, new_factors)
            if quality_factors.empty:
                logger.warning("Factor-level IC skipped: cannot map factor tasks to factor columns.")
                return

            quality_by_factor = self.calculate_factor_level_ic(quality_factors)
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

    def calculate_factor_level_ic(self, factors: pd.DataFrame) -> dict[str, dict[str, float]]:
        factor_df = factors.copy()
        factor_df.columns = self.flatten_columns(factor_df.columns)
        factor_df = self.normalize_factor_index(factor_df)
        label = self.load_qlib_label(factor_df.index)
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
                if len(group) < 2:
                    daily_rows.append((np.nan, np.nan))
                    continue
                factor_std = group["factor"].std()
                label_std = group["label"].std()
                if pd.isna(factor_std) or pd.isna(label_std) or factor_std == 0 or label_std == 0:
                    daily_rows.append((np.nan, np.nan))
                    continue
                daily_rows.append(
                    (
                        group["factor"].corr(group["label"]),
                        group["factor"].rank().corr(group["label"].rank()),
                    )
                )

            daily_ic = pd.DataFrame(daily_rows, columns=["IC", "Rank IC"])
            results[column] = {
                "IC": float(daily_ic["IC"].mean()) if daily_ic["IC"].notna().any() else np.nan,
                "Rank IC": float(daily_ic["Rank IC"].mean()) if daily_ic["Rank IC"].notna().any() else np.nan,
            }
        return results

    def load_qlib_label(self, index: pd.Index) -> pd.Series:
        datetime_level = self.get_index_level_name(index, ("datetime",))
        instrument_level = self.get_index_level_name(index, ("instrument", "code", "symbol"))
        if datetime_level is None or instrument_level is None:
            raise ValueError(f"factor index must include datetime and instrument levels, got {index.names}")

        dates = index.get_level_values(datetime_level)
        instruments = sorted({str(inst) for inst in index.get_level_values(instrument_level)})
        start_time = pd.Timestamp(dates.min()).strftime("%Y-%m-%d")
        end_time = pd.Timestamp(dates.max()).strftime("%Y-%m-%d")

        import qlib
        from qlib.config import REG_CN
        from qlib.data import D

        provider_uri = os.environ.get("QLIB_PROVIDER_URI", "~/.qlib/qlib_data/cn_data")
        qlib.init(provider_uri=str(Path(provider_uri).expanduser()), region=REG_CN)
        label_df = D.features(
            instruments,
            ["Ref($close, -2)/Ref($close, -1) - 1"],
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
