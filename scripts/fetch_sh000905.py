"""Fetch SH000905 (CSI 500 index) daily OHLCV via baostock with retries, save in qlib raw_data schema."""
import time
import baostock as bs
import pandas as pd
from pathlib import Path

OUT_DIR = Path.home() / ".qlib/qlib_data/cn_data/raw_data_back_adjust"
OUT_DIR.mkdir(parents=True, exist_ok=True)
START = "2014-12-31"
END = "2026-04-08"

FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"

df = None
for attempt in range(8):
    print(f"=== attempt {attempt + 1}/8 ===")
    try:
        login = bs.login()
        print(f"login: error_code={login.error_code}, msg={login.error_msg}")
        if login.error_code != "0":
            time.sleep(5)
            continue
        rs = bs.query_history_k_data_plus(
            "sh.000905", FIELDS,
            start_date=START, end_date=END,
            frequency="d", adjustflag="3",
        )
        print(f"query: error_code={rs.error_code}, msg={rs.error_msg}")
        tmp = rs.get_data()
        bs.logout()
        if not tmp.empty:
            df = tmp
            print(f"got {len(df)} rows")
            break
        print("empty result; will retry")
    except Exception as e:
        print(f"exception: {e}")
        try: bs.logout()
        except: pass
    time.sleep(5 * (attempt + 1))

assert df is not None and not df.empty, "all attempts failed"

df["code"] = "sh000905"
df["factor"] = "1.0"
out = OUT_DIR / "sh000905.csv"
df.to_csv(out, index=False)
print(f"\nSaved {len(df)} rows to {out}")
print(df.head(2).to_string(index=False))
print(df.tail(2).to_string(index=False))
