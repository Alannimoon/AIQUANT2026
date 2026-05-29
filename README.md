# AIQUANT2026 — ELITEALPHA Project

NeurIPS 2026 submission: **ELITEALPHA — Illuminating the Alpha Factor Space with MAP-Elites**.

Tsinghua CS Dept · co-authors Ruili Liu + Yiwei Jiang.

## What's in this repo

```
AIQUANT2026/
├── AlphaAgent/                       # Patched AlphaAgent (KDD 2025) — baseline 1
├── elitealpha/                       # ELITEALPHA main code  (to be written)
├── docs/
│   ├── elitealpha_proposal.pdf       # original project proposal
│   └── AlphaAgent_walkthrough.md     # how AlphaAgent works + our patches
├── scripts/
│   ├── fetch_sh000905.py             # baostock fetcher (network can be flaky)
│   └── fetch_sh000905_ak.py          # akshare fetcher (Sina endpoint, more reliable)
└── README.md                         # this file
```

## AlphaAgent baseline status

- Forked from upstream commit `1da96e94` (`RndmVariableQ/AlphaAgent`, KDD 2025)
- 7 local patches applied for DeepSeek + local CN data adaptation. See `docs/AlphaAgent_walkthrough.md` §7
- One full Loop 0 with portfolio metrics recorded (IC=0.062, Rank IC=0.0396, excess AR=-9.54%)

## Quick start (new team member)

```bash
# 1. Create conda env
conda create -n alphaagent python=3.10 -y
conda activate alphaagent
cd AlphaAgent
pip install -e .

# 2. Configure LLM
cp .env.example .env
# Edit .env: set OPENAI_BASE_URL=https://api.deepseek.com, OPENAI_API_KEY=<your_key>,
#           CHAT_MODEL=deepseek-chat, REASONING_MODEL=deepseek-chat

# 3. Ensure Qlib CN data is in ~/.qlib/qlib_data/cn_data/
# (auto-downloads on first qrun if missing, or use AlphaAgent/prepare_cn_data.py)

# 4. (One-time) Add SH000905 benchmark — Sina endpoint is most reliable
conda activate base  # need pandas>=2.0 for akshare
env -u http_proxy -u https_proxy -u all_proxy python scripts/fetch_sh000905_ak.py
# Then dump it into qlib data:
mkdir -p /tmp/sh000905_dump && cp ~/.qlib/qlib_data/cn_data/raw_data_back_adjust/sh000905.csv /tmp/sh000905_dump/
# Need qlib source for dump_bin.py:
git clone --depth 1 https://github.com/microsoft/qlib.git
conda run -n alphaagent python qlib/scripts/dump_bin.py dump_update \
  --data_path /tmp/sh000905_dump \
  --qlib_dir ~/.qlib/qlib_data/cn_data \
  --include_fields open,high,low,close,preclose,volume,amount,turn,factor \
  --symbol_field_name code --date_field_name date

# 5. Run AlphaAgent baseline
cd AlphaAgent
conda run -n alphaagent --no-capture-output alphaagent mine \
    --potential_direction "your market hypothesis" \
    > run_logs/run_NNN.log 2>&1 &
tail -F run_logs/run_NNN.log
```

## Important caveats for collaborators

- **Never commit `.env`** — contains DeepSeek API key. Already gitignored.
- **Caches are gitignored** (`AlphaAgent/git_ignore_folder/`, `pickle_cache/`, `log/`, `run_logs/`). Each member's caches are local.
- **AlphaAgent's RAG / knowledge graph is stubbed** (3 of our 7 patches) because DeepSeek has no embedding API. Don't be confused by what looks like "RAG retrieval" — it's a no-op.
- **`conf.yaml` and `conf_cn_combined_kdd_ver.yaml` have PortAnaRecord enabled** (re-enabled after we got SH000905). If you see `LoadObjectError: report_normal_1day.pkl`, check that SH000905 is in your `~/.qlib/qlib_data/cn_data`.
- **Do not modify `AlphaAgent/.env.example`** — keep it pristine for upstream parity.

## Read this before coding ELITEALPHA

`docs/AlphaAgent_walkthrough.md` covers how the 5-step loop works, what gets called, where artifacts land. Read §3 (execution flow) and §10 (integration points for MAP-Elites) before writing any new code.

## Upstream

AlphaAgent at commit `1da96e94a06a925c3997899f1848899440585efe`:
https://github.com/RndmVariableQ/AlphaAgent

To pull upstream updates later:
```bash
cd AlphaAgent
git init && git remote add upstream https://github.com/RndmVariableQ/AlphaAgent.git
git fetch upstream main
# manually merge what you want
```
