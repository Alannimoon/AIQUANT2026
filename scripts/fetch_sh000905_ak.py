"""Try multiple akshare endpoints to fetch SH000905 (CSI 500 index)."""
import time
import pandas as pd
import akshare as ak
from pathlib import Path

OUT_DIR = Path.home() / ".qlib/qlib_data/cn_data/raw_data_back_adjust"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_sina(df):
    """Sina format: date,open,high,low,close,volume."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[["date", "open", "high", "low", "close", "volume"]]

def normalize_em(df):
    """eastmoney 日,开,收,高,低,成交量,成交额,...."""
    df = df.copy()
    cols = {"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low",
            "成交量":"volume","成交额":"amount","换手率":"turn","涨跌幅":"pctChg"}
    df = df.rename(columns=cols)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    keep = ["date","open","high","low","close","volume"]
    for c in ["amount","turn","pctChg"]:
        if c in df.columns: keep.append(c)
    return df[keep]

attempts = [
    ("stock_zh_index_daily (sina)", lambda: ak.stock_zh_index_daily(symbol="sh000905"), normalize_sina),
    ("stock_zh_index_daily_tx (tencent)", lambda: ak.stock_zh_index_daily_tx(symbol="sh000905"), normalize_sina),
    ("index_zh_a_hist (eastmoney)", lambda: ak.index_zh_a_hist(symbol="000905", period="daily", start_date="20141231", end_date="20260408"), normalize_em),
]

df = None
for name, fn, normalize in attempts:
    print(f"=== trying {name} ===")
    try:
        raw = fn()
        print(f"  ok: {len(raw)} rows, cols={list(raw.columns)}")
        df = normalize(raw)
        print(f"  normalized: {len(df)} rows")
        print(df.head(2).to_string(index=False))
        break
    except Exception as e:
        print(f"  fail: {type(e).__name__}: {e}")
        time.sleep(2)

assert df is not None, "all akshare endpoints failed"

df["code"] = "sh000905"
df["preclose"] = df["close"].shift(1).fillna(df["close"])
df["tradestatus"] = 1
for c in ["amount","turn","pctChg"]:
    if c not in df.columns: df[c] = ""
df["peTTM"] = ""
df["pbMRQ"] = ""
df["psTTM"] = ""
df["pcfNcfTTM"] = ""
df["isST"] = 0
df["factor"] = 1.0

cols = ["date","code","open","high","low","close","preclose","volume","amount","turn",
        "tradestatus","pctChg","peTTM","pbMRQ","psTTM","pcfNcfTTM","isST","factor"]
df = df[cols]
out = OUT_DIR / "sh000905.csv"
df.to_csv(out, index=False)
print(f"\nSaved {len(df)} rows to {out}")
print(df.head(2).to_string(index=False))
print(df.tail(2).to_string(index=False))
