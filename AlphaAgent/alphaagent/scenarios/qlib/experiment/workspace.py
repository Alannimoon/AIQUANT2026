from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from alphaagent.core.experiment import FBWorkspace
from alphaagent.log import logger
from alphaagent.utils.env import QTDockerEnv


class QlibFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.template_folder_path = template_folder_path
        self.inject_code_from_folder(template_folder_path)

    def execute(
        self, 
        qlib_config_name: str = "conf.yaml", 
        run_env: dict = {}, 
        use_local: bool = True, 
        *args, 
        **kwargs
    ) -> str:
        # 使用本地环境或Docker环境
        qtde = QTDockerEnv(is_local=use_local)
        qtde.prepare()

        template_folder_path = getattr(self, "template_folder_path", None)
        result_reader = template_folder_path / "read_exp_res.py" if template_folder_path is not None else None
        if result_reader is not None and result_reader.exists():
            shutil.copy2(result_reader, self.workspace_path / "read_exp_res.py")
        
        # 运行Qlib回测
        logger.info(f"Execute {'Local' if use_local else 'Docker container'} Backtest: qrun {qlib_config_name}")
        execute_log = qtde.run(
            local_path=str(self.workspace_path),
            entry=f"qrun {qlib_config_name}",
            env=run_env,
        )

        # 处理结果
        logger.info(f"Read {'Local' if use_local else 'Docker container'} Backtest Result")
        execute_log = qtde.run(
            local_path=str(self.workspace_path),
            entry="python read_exp_res.py",
            env=run_env,
        )

        # 加载结果
        ret_pkl = self.workspace_path / "ret.pkl"
        if ret_pkl.exists():
            ret_df = pd.read_pickle(ret_pkl)
            logger.log_object(ret_df, tag="Quantitative Backtesting Chart")

        csv_path = self.workspace_path / "qlib_res.csv"
        if not csv_path.exists():
            logger.error(f"File {csv_path} does not exist.")
            return None

        result = pd.read_csv(csv_path, index_col=0).iloc[:, 0]
        nan_metrics = result[result.isna()]
        if not nan_metrics.empty:
            logger.warning(f"Qlib metrics contain NaN values:\n{nan_metrics}")

        ic_debug_summary_path = self.workspace_path / "ic_debug_summary.csv"
        if ic_debug_summary_path.exists():
            ic_debug_summary = pd.read_csv(ic_debug_summary_path)
            logger.info(f"IC debug summary:\n{ic_debug_summary.to_string(index=False)}")
            logger.log_object(ic_debug_summary, tag="IC Debug Summary")

        ic_debug_by_date_path = self.workspace_path / "ic_debug_by_date.csv"
        if ic_debug_by_date_path.exists():
            ic_debug_by_date = pd.read_csv(ic_debug_by_date_path)
            if "bad_reason" in ic_debug_by_date:
                bad_dates = ic_debug_by_date[ic_debug_by_date["bad_reason"].fillna("").astype(str).ne("")]
            else:
                bad_dates = ic_debug_by_date.iloc[0:0]
            logger.info(
                "IC debug by date: "
                f"rows={len(ic_debug_by_date)}, bad_dates={len(bad_dates)}"
            )
            if not bad_dates.empty:
                logger.info(f"IC debug bad date examples:\n{bad_dates.head(10).to_string(index=False)}")
            logger.log_object(ic_debug_by_date, tag="IC Debug By Date")

        return result
