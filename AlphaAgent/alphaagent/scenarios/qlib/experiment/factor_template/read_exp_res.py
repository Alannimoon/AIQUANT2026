import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

qlib.init()

from qlib.workflow import R

# here is the documents of the https://qlib.readthedocs.io/en/latest/component/recorder.html

# TODO: list all the recorder and metrics

def write_ic_debug(recorder, output_dir: Path) -> None:
    pred = recorder.load_object("pred.pkl")
    label = recorder.load_object("label.pkl")

    pred_series = first_numeric_series(pred, "pred")
    label_series = first_numeric_series(label, "label")
    aligned = pd.concat([pred_series, label_series], axis=1, join="inner").replace([np.inf, -np.inf], np.nan)
    aligned.columns = ["pred", "label"]
    aligned = aligned.dropna()

    if aligned.empty:
        summary = pd.DataFrame(
            [
                {
                    "rows": 0,
                    "dates": 0,
                    "reason": "pred/label overlap is empty after dropping NaN/inf",
                }
            ]
        )
        summary.to_csv(output_dir / "ic_debug_summary.csv", index=False)
        return

    if "datetime" not in aligned.index.names:
        summary = pd.DataFrame(
            [
                {
                    "rows": len(aligned),
                    "dates": None,
                    "reason": f"index has no datetime level: {aligned.index.names}",
                }
            ]
        )
        summary.to_csv(output_dir / "ic_debug_summary.csv", index=False)
        return

    rows = []
    for dt, group in aligned.groupby(level="datetime"):
        pred_valid = group["pred"].dropna()
        label_valid = group["label"].dropna()
        pair = group[["pred", "label"]].dropna()
        pred_std = pair["pred"].std()
        label_std = pair["label"].std()
        enough_pairs = len(pair) >= 2
        pred_varies = pd.notna(pred_std) and pred_std != 0
        label_varies = pd.notna(label_std) and label_std != 0
        rows.append(
            {
                "datetime": dt,
                "pair_count": int(len(pair)),
                "pred_count": int(len(pred_valid)),
                "label_count": int(len(label_valid)),
                "pred_unique": int(pair["pred"].nunique(dropna=True)),
                "label_unique": int(pair["label"].nunique(dropna=True)),
                "pred_std": pred_std,
                "label_std": label_std,
                "pearson_ic": pair["pred"].corr(pair["label"]) if enough_pairs and pred_varies and label_varies else np.nan,
                "rank_ic": (
                    pair["pred"].rank().corr(pair["label"].rank())
                    if enough_pairs and pred_varies and label_varies
                    else np.nan
                ),
                "bad_reason": bad_ic_reason(len(pair), pred_std, label_std),
            }
        )

    debug = pd.DataFrame(rows)
    debug.to_csv(output_dir / "ic_debug_by_date.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "rows": int(len(aligned)),
                "dates": int(len(debug)),
                "valid_pearson_dates": int(debug["pearson_ic"].notna().sum()),
                "valid_rank_dates": int(debug["rank_ic"].notna().sum()),
                "bad_dates": int(debug["bad_reason"].ne("").sum()),
                "mean_pearson_ic": debug["pearson_ic"].mean(),
                "mean_rank_ic": debug["rank_ic"].mean(),
                "pred_zero_std_dates": int((debug["pred_std"] == 0).sum()),
                "label_zero_std_dates": int((debug["label_std"] == 0).sum()),
                "too_few_pair_dates": int((debug["pair_count"] < 2).sum()),
            }
        ]
    )
    summary.to_csv(output_dir / "ic_debug_summary.csv", index=False)
    print("IC debug summary:")
    print(summary.T)


def first_numeric_series(obj, name: str) -> pd.Series:
    if isinstance(obj, pd.Series):
        series = obj
    elif isinstance(obj, pd.DataFrame):
        numeric_cols = obj.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            raise ValueError(f"{name} has no numeric columns")
        series = obj[numeric_cols[0]]
    else:
        raise TypeError(f"{name} must be Series or DataFrame, got {type(obj)!r}")
    return pd.to_numeric(series, errors="coerce").rename(name)


def bad_ic_reason(pair_count: int, pred_std, label_std) -> str:
    reasons = []
    if pair_count < 2:
        reasons.append("too_few_pairs")
    if pd.isna(pred_std):
        reasons.append("pred_std_nan")
    elif pred_std == 0:
        reasons.append("pred_zero_std")
    if pd.isna(label_std):
        reasons.append("label_std_nan")
    elif label_std == 0:
        reasons.append("label_zero_std")
    return "|".join(reasons)


def find_latest_recorder():
    experiments = R.list_experiments()
    latest_recorder = None
    for experiment in experiments:
        recorders = R.list_recorders(experiment_name=experiment)
        for recorder_id in recorders:
            if recorder_id is not None:
                recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
                end_time = recorder.info["end_time"]
                if latest_recorder is None or end_time > latest_recorder.info["end_time"]:
                    latest_recorder = recorder
    return latest_recorder


def main() -> None:
    latest_recorder = find_latest_recorder()
    if latest_recorder is None:
        print("No recorders found")
        return

    print(f"Latest recorder: {latest_recorder}")

    metrics = pd.Series(latest_recorder.list_metrics())
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "qlib_res.csv"
    metrics.to_csv(output_path)

    print(f"Output has been saved to {output_path}")
    print("NaN metrics:")
    print(metrics[metrics.isna()])

    try:
        write_ic_debug(latest_recorder, output_dir)
    except Exception as e:
        print(f"Failed to write IC debug files: {e}")

    ret_data_frame = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    ret_data_frame.to_pickle("ret.pkl")


if __name__ == "__main__":
    main()
