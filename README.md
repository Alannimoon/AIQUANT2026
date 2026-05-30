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
│   ├── fetch_sh000905.py             # DEPRECATED: chenditc bundles SH000905. Kept for reference only.
│   └── fetch_sh000905_ak.py          # DEPRECATED: same as above.
└── README.md                         # this file
```

## AlphaAgent baseline status

- Forked from upstream commit `1da96e94` (`RndmVariableQ/AlphaAgent`, KDD 2025)
- 7 local patches applied for DeepSeek + chenditc data adaptation. See `docs/AlphaAgent_walkthrough.md` §7
- End-to-end mine + backtest pipeline verified; full portfolio metrics (IC / RankIC / AR / IR / MDD) produced per loop
- Multi-loop baseline numbers TBD — single-loop runs have high variance; will collect distribution across 5+ full runs

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

# 3. Download Qlib CN data from chenditc/investment_data (recommended source)
#    ~550 MB tarball, extracts to ~3 GB. Includes proper CSI500 universe membership
#    history (1774 codes across windows), SH000905 benchmark, vwap/adjclose fields.
#    See https://github.com/chenditc/investment_data/releases for the latest release tag.
LATEST_TAG=$(curl -sL https://api.github.com/repos/chenditc/investment_data/releases/latest | grep -oP '"tag_name":\s*"\K[^"]+')
wget -O /tmp/qlib_bin.tar.gz https://github.com/chenditc/investment_data/releases/download/${LATEST_TAG}/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxf /tmp/qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
rm /tmp/qlib_bin.tar.gz
# Verify:
ls ~/.qlib/qlib_data/cn_data/   # should show: calendars  features  instruments

# 4. Run AlphaAgent baseline
cd AlphaAgent
conda run -n alphaagent --no-capture-output alphaagent mine \
    --potential_direction "your market hypothesis" \
    > run_logs/run_NNN.log 2>&1 &
tail -F run_logs/run_NNN.log
```

## Important caveats for collaborators

- **Never commit `.env`** — contains DeepSeek API key. Already gitignored.
- **Caches are gitignored** (`AlphaAgent/git_ignore_folder/`, `pickle_cache/`, `log/`, `run_logs/`). Each member's caches are local.
- **AlphaAgent's RAG knowledge base is silently disabled** via `knowledge_self_gen=False` in `alphaagent_loop.py:73`. This keeps the CoSTEER knowledge base empty so `calculate_embedding_distance_between_str_list` short-circuits (no /v1/embeddings call needed). Don't be confused by what looks like "RAG retrieval" — it's a no-op for now.
- **`conf.yaml` and `conf_cn_combined_kdd_ver.yaml` have PortAnaRecord enabled**. The benchmark SH000905 ships with the chenditc data tarball. If you see `LoadObjectError: report_normal_1day.pkl`, check that SH000905 is in `~/.qlib/qlib_data/cn_data/features/`.
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
