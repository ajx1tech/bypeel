<div align="center">

# 🔗 bypeel

### AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

*Fusing network-layer telemetry with blockchain-layer data into one explainable, offline investigative platform.*

**Smart India Hackathon 2026 · Problem Statement 26146**
**Organisation:** National Technical Research Organisation (NTRO) · **Theme:** Blockchain & Cybersecurity · **Team:** Endeavour

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![Offline](https://img.shields.io/badge/Runtime-100%25%20Offline-2E8B57)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-informational)
![License](https://img.shields.io/badge/License-MIT-blue)
![Status](https://img.shields.io/badge/Status-Working%20Prototype-yellow)

</div>

---

## 📖 Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Why This Is Hard](#2-why-this-is-hard)
3. [What You Get](#3-what-you-get)
4. [What Makes bypeel Different](#4-what-makes-bypeel-different)
5. [Architecture](#5-architecture)
6. [Tech Stack](#6-tech-stack)
7. [Repository Layout](#7-repository-layout)
8. [Installation and Setup](#8-installation-and-setup)
9. [Using It](#9-using-it)
10. [Dataset and Schema](#10-dataset-and-schema)
11. [Methodology](#11-methodology)
12. [Explainability](#12-explainability)
13. [The Local Database](#13-the-local-database)
14. [Problem-Statement Requirement Mapping](#14-problem-statement-requirement-mapping)
15. [Existing Solutions and Prior Art](#15-existing-solutions-and-prior-art)
16. [Retraining the Model](#16-retraining-the-model)
17. [Running Fully Air-Gapped](#17-running-fully-air-gapped)
18. [Troubleshooting](#18-troubleshooting)
19. [Roadmap](#19-roadmap)
20. [Team](#20-team)

---

## 1. Problem Statement

Bitcoin's pseudonymous, peer-to-peer design lets criminal actors move, layer,
and cash out illicit funds — ransomware payments, darknet-market proceeds,
extortion, and laundering — while evading traditional financial surveillance.

Every Bitcoin transaction is permanently recorded on a public ledger, but
wallet addresses aren't tied to real identities. The one structural crack in
that anonymity: to broadcast a transaction, a node connects to the Bitcoin
P2P network over a real IP address — a rare, fleeting link between an
on-chain identity (a wallet) and a real-world signal (a network location).

PS 26146 (NTRO) asks for a complete **offline** system that:

1. **Ingests** bulk Bitcoin transaction/network metadata (CSV/JSON/XML)
2. **Correlates** network-layer observations (IP/port/timing) with
   blockchain-layer data (wallet/TXID/amount)
3. **Applies AI/ML** — a working model, not just rules — to detect
   anomalies, cluster entities, and generate prioritized, explainable
   investigative leads
4. **Visualizes** findings via a dashboard / link-analysis interface,
   fully offline, on Linux

bypeel is our complete answer to that brief — see §14 for a line-by-line
mapping of every requirement to the exact file that satisfies it.

---

## 2. Why This Is Hard

| Challenge | Why it matters |
|---|---|
| **No ground truth** | We rarely know for certain which wallets are criminal — this is fundamentally closer to unsupervised anomaly detection than clean classification. |
| **Evasion techniques** | Mixers, CoinJoin, peel chains, and one-time addresses all look statistically "weird" even when innocent, which is exactly how false positives happen. |
| **Noisy network signals** | An observed IP might be a VPN exit node, Tor relay, or an innocent peer just relaying traffic for someone else — not the true broadcaster. |
| **The explainability requirement** | An agency can't act on a black-box score — every flag has to be justifiable, in plain language, to a human investigator and eventually a court. |
| **The offline constraint** | No live blockchain node, no cloud APIs, no CDN — GeoIP, the ML model, and every UI library have to run from local disk. |

---

## 3. What You Get

- A **Python FastAPI backend** running a real, trained **IsolationForest**
  (persisted via `joblib`), **exact SHAP `TreeExplainer`** attributions, a
  **NetworkX** heterogeneous graph, and **Union-Find** entity resolution
  (Common-Input-Ownership Heuristic).
- An **interactive dashboard** — Overview (KPIs, charts, live pipeline
  view), Graph Explorer (node-link canvas *and* a dedicated Entity Cluster
  bubble map), a ranked Alerts queue with SHAP-style explainability cards,
  and a Dossier & Export tab with SHA-256 chain-of-custody sealing.
- **Offline GeoIP/ASN enrichment** via a bundled MaxMind GeoLite2 `.mmdb` —
  zero network calls, zero API keys.
- **Local SQLite persistence** (`data/bypeel.db`) — every dataset you
  process is saved as a "case" on your own machine and can be reopened
  instantly without re-running the pipeline. No cloud, no external DB
  server, no telemetry.
- A **client-side JS fallback engine** so the dashboard still works with
  zero installation (open the HTML file directly) for a 30-second
  screening demo, with the full Python pipeline as the default, primary
  path once the server is running.

---

## 4. What Makes bypeel Different

The core building blocks for this problem — graph analysis, anomaly
detection, entity clustering, explainability — are the *correct* tools, and
any competent submission converges on them. Where bypeel differentiates:

**1. Verifiably offline, not just offline-labeled.** Every JS library
(D3, Chart.js, PapaParse) is vendored locally under `static/vendor/` — zero
CDN, checkable in a browser's Network tab. This matters because it's easy
to *claim* "air-gapped" in a README while actually shipping a cloud-deployed
app with a localhost dev mode; bypeel's architecture makes the claim
literally true and easy to demonstrate live.

**2. A real trained model wired end-to-end, not a placeholder.** The
`IsolationForest` in `models/isolation_forest.pkl` is loaded via `joblib`
and scored per-transaction in `app/ml_engine.py`; every alert's
explanation comes from an exact SHAP `TreeExplainer` pass over that same
model, not a cosmetic bar chart.

**3. Two working pipelines, same output contract.** The Python backend
(the "real" system) and an in-browser JS engine both produce the identical
JSON shape the dashboard renders, so the tool degrades gracefully to a
zero-install demo instead of failing outright if a judge's machine can't
run the server.

**4. A local case database, not just a one-shot script.** Processed
datasets persist to SQLite and reopen instantly — a small but real product
decision most hackathon-stage prototypes skip.

**5. Honest about its own limits.** §18 and §19 name real gaps (see §15 for
a full comparison against other public submissions for this same problem
statement) rather than overstating coverage.

---

## 5. Architecture

```
 ┌───────────────────────────────────────────────────────────────────────┐
 │  Dashboard — static/bypeel.html  (all libraries vendored locally)       │
 │  static/vendor/d3.min.js · chart.umd.min.js · papaparse.min.js          │
 │                                                                         │
 │  Ingest tab                                                             │
 │    ├─ "Load Demo Dataset"  → bundled 10,126-tx synthetic corpus         │
 │    ├─ "Saved Cases" panel  → GET /api/cases  (reload from SQLite)       │
 │    └─ "Process Dataset"    → POST /api/process (Python backend, default)│
 │                                or in-browser JS engine (no server needed)│
 │                                                                         │
 │  Overview · Graph Explorer (node-link + Entity Clusters) · Alerts ·     │
 │  Dossier & Export                                                       │
 └────────────────────────────┬────────────────────────────────────────────┘
                               │ HTTP, localhost only
                               ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │  Backend — app/  (FastAPI, Python 3.11-3.13)                            │
 │                                                                         │
 │  ingest.py        CSV / JSON / XML → normalized pandas DataFrame        │
 │  geoip_lookup.py  Offline MaxMind GeoLite2 (.mmdb) country/ASN enrich   │
 │  features.py      17-feature engineering (matches trained model)        │
 │  graph_engine.py  NetworkX heterogeneous graph + Union-Find (CIOH)      │
 │  ml_engine.py     joblib-persisted IsolationForest + SHAP explainer     │
 │  motifs.py        Rule-based laundering motif detection                 │
 │  analytics.py     Peel-chain hop tracing + Overview KPI builder         │
 │  pipeline.py       Orchestrates all of the above → one JSON bundle      │
 │  db.py             SQLite persistence (cases, dossiers)                 │
 │  main.py           FastAPI app: serves UI + /api/* endpoints            │
 └────────────────────────────┬────────────────────────────────────────────┘
                               ▼
                    data/bypeel.db  (SQLite, single file, your machine)
                    data/geoip/GeoLite2-Country.mmdb
                    models/isolation_forest.pkl
```

**Why two pipelines?** The Python backend is the primary, default path — a
trained model, exact SHAP attributions, a full NetworkX graph, and
persistent local storage. The in-browser JS engine is kept as a zero-install
fallback so the dashboard still functions as a standalone artifact if
someone opens the HTML file without ever starting the server.

---

## 6. Tech Stack

| Layer | Tools | Pinned version |
|---|---|---|
| Language | Python (64-bit) | 3.11 – 3.13 |
| Data handling | pandas, NumPy | 2.2.3, 2.2.6 |
| Storage | SQLite (stdlib `sqlite3`) | bundled with Python |
| GeoIP | MaxMind GeoLite2 (offline `.mmdb`) + `geoip2` | 4.8.0 |
| Graph | NetworkX (Union-Find for CIOH entity resolution) | 3.4.2 |
| Anomaly detection | scikit-learn `IsolationForest` | **1.9.0** (matches the bundled model) |
| Explainability | SHAP (`TreeExplainer`), numba, llvmlite | 0.48.0, 0.61.2, 0.44.0 |
| Backend API | FastAPI + Uvicorn + python-multipart | 0.115.6, 0.32.1, 0.20 |
| Frontend | Vanilla JS + D3.js (graph) + Chart.js (charts) + PapaParse (CSV) — all vendored, zero CDN | files in `static/vendor/` |
| Model persistence | `joblib` | 1.4.2 |
| Report export | Browser-native print-to-PDF (chain-of-custody hashing via Web Crypto `SHA-256`); `reportlab` available for a server-rendered alternative | 4.2.5 |

All versions are pinned in `requirements.txt` (direct) and `requirements.lock.txt` (complete) — see §8.7.

---

## 7. Repository Layout

```
bypeel/
├── app/
│   ├── __init__.py
│   ├── main.py             FastAPI app: UI + /api/process, /api/cases, /api/dossiers, /api/health
│   ├── db.py                SQLite persistence layer (cases + dossiers)
│   ├── pipeline.py          Orchestrator: ingest → geoip → features → ML → motifs → graph
│   ├── ingest.py            CSV/JSON/XML parsing + column normalization
│   ├── geoip_lookup.py      Offline GeoIP/ASN enrichment (MaxMind GeoLite2)
│   ├── features.py          17-feature engineering
│   ├── graph_engine.py      NetworkX graph + Union-Find CIOH entity resolution
│   ├── ml_engine.py         IsolationForest scoring + SHAP explainability
│   ├── motifs.py            Rule-based laundering motif detection
│   └── analytics.py         Peel-chain tracing + Overview KPI builder
├── models/
│   └── isolation_forest.pkl    Pre-trained model bundle (model+scaler+feature_names)
├── data/
│   ├── geoip/
│   │   ├── GeoLite2-Country.mmdb
│   │   ├── COPYRIGHT.txt
│   │   └── LICENSE.txt
│   └── bypeel.db                Created automatically on first run (git-ignored)
├── static/
│   ├── bypeel.html               The frontend dashboard
│   └── vendor/                   Locally vendored JS — NO CDN
│       ├── d3.min.js
│       ├── chart.umd.min.js
│       └── papaparse.min.js
├── sample_data/
│   ├── labeled_transactions.csv
│   ├── normal_transactions.csv
│   └── ranked_alerts.csv
├── scripts/
│   └── train_model.py       Retrain the IsolationForest on your own data
├── requirements.txt          Direct dependencies, exact pins
├── requirements.lock.txt     Every package pinned, wheels-only, Windows/Linux/macOS, Python 3.11-3.13
├── run.sh                    One-command setup + launch (Linux/macOS)
├── run.bat                   One-command setup + launch (Windows)
├── wheelhouse/               OPTIONAL: pre-downloaded wheels for offline installs (see §17; git-ignored)
├── .gitattributes            Keeps run.sh as LF and run.bat as CRLF line endings
├── .gitignore
└── README.md
```

> **Note on the frontend file layout.** `static/bypeel.html` currently
> ships as a single self-contained file by design (it's a genuine strength
> for portability — the entire dashboard is one artifact that also runs
> standalone with zero server). We're in the process of splitting it into
> a conventional `static/dashboard/{index.html, css/, js/}` structure for
> easier navigation/maintenance in the repo's file tree, without changing
> any behavior — see the open item in §19.

**What to push to GitHub:** everything above, as-is. `models/isolation_forest.pkl`
(~2.5 MB), `data/geoip/GeoLite2-Country.mmdb` (~8.6 MB), and the vendored JS
(~500 KB total) are all small enough for a normal `git push` — no Git LFS
needed. `data/bypeel.db` is git-ignored by design: it's per-installation
local case storage, not source code.

---

## 8. Installation and Setup

> **For judges and evaluators.** Follow **one** track — Linux (§8.3) *or* Windows (§8.4).
> Each ends with the dashboard open in your browser. Expect about 5 minutes: roughly
> **145 MB** of Python packages are downloaded **once**, and from then on bypeel never
> touches the network again (a fully offline method is in §17).

### 8.1 What you need

| | Requirement |
|---|---|
| **Python** | **3.11, 3.12 or 3.13 — 64-bit.** Recommended: **3.12**. |
| **OS** | Windows 10/11 (x64) · Ubuntu 22.04 / 24.04 · Debian 12 · other mainstream x86-64 Linux · macOS (wheels resolve; see matrix below) |
| **Disk / RAM** | ≈ 650 MB for the virtual environment + ≈ 15 MB for the repo · 4 GB RAM |
| **Network** | Only once, so `pip` can fetch packages. Not needed to run. |
| **C/C++ compiler** | **Not needed.** Every dependency installs from a pre-built wheel; the scripts pass `--only-binary=:all:`, so a missing wheel fails in seconds with a short message instead of a long compiler traceback. |
| **Git** | Optional — you can download the repository as a ZIP instead. |

**Why not Python 3.10 or 3.14?** The bundled model (`models/isolation_forest.pkl`) was pickled
with **scikit-learn 1.9.0**, and bypeel pins exactly that version so the model loads with no
warnings. scikit-learn 1.9.x publishes wheels only for Python ≥ 3.11, and the pinned
`shap`/`numba` stack only goes up to 3.13. So the supported window is **3.11 – 3.13**.
Ubuntu 22.04 ships Python 3.10 — the Linux track below shows how to add 3.12 beside it.

### 8.2 Check your Python first (30 seconds)

Run this in any terminal (Linux, macOS, Windows PowerShell or CMD):

```bash
python3 -c "import sys,struct; print(sys.version.split()[0], str(struct.calcsize('P')*8)+'-bit')"
```
On Windows use `python` instead of `python3` (or `py -3.12`). You need to see
`3.11.x`, `3.12.x` or `3.13.x` **and** `64-bit`. If not, do Step 1 of your track.
You can also skip this check: `run.sh` / `run.bat` search for a suitable Python themselves
and tell you exactly what to install if they find none.

### 8.3 Linux track (Ubuntu / Debian / Fedora / others)

**Step 1 — install system packages**

```bash
# Ubuntu 24.04 (Python 3.12) and Debian 12 (Python 3.11):
sudo apt update && sudo apt install -y git python3 python3-venv python3-pip

# Ubuntu 22.04 (ships Python 3.10 — too old; install 3.12 alongside it):
sudo apt update && sudo apt install -y git software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update && sudo apt install -y python3.12 python3.12-venv

# Fedora:
sudo dnf install -y git python3.12
```
*Any other distro, or no root access?* Install the `uv` tool and let it fetch Python 3.12:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
uv python install 3.12
```

**Step 2 — get the code**

```bash
git clone https://github.com/ajx1tech/bypeel.git
cd bypeel
ls      # you should see: app  data  models  sample_data  static  run.sh  requirements.lock.txt ...
```

**Step 3 — run it**

```bash
chmod +x run.sh
./run.sh
```
Python 3.12 installed beside the system 3.10, or via `uv`? Tell the script which one:
```bash
PYTHON=python3.12 ./run.sh                 # apt / deadsnakes
PYTHON="$(uv python find 3.12)" ./run.sh   # uv
```
Expected output (first run takes 1–3 minutes, mostly the download):
```
[bypeel] Using Python 3.12.x (/usr/bin/python3.12)
[bypeel] Creating virtual environment...
[bypeel] Installing dependencies (first run needs internet; afterwards 100% offline)...
[bypeel] Verifying model + GeoIP database load correctly...
  Model + GeoIP OK
[bypeel] Starting server on http://127.0.0.1:8000  (Ctrl+C to stop)
```

**Step 4 — open <http://127.0.0.1:8000>** in a browser, then continue with §8.5.

<details>
<summary><b>Prefer typing the commands yourself?</b> (exactly what <code>run.sh</code> does)</summary>

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --only-binary=:all: -r requirements.lock.txt
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
</details>

### 8.4 Windows track (Windows 10 / 11, 64-bit)

**Step 1 — install Python 3.12 (64-bit) and Git**

Open **PowerShell** and run:
```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Git.Git
```
No `winget`? Download the **Windows installer (64-bit)** from <https://www.python.org/downloads/windows/>
and **tick "Add python.exe to PATH"** on the first screen of the installer.
**Close PowerShell and open a new window** (so PATH is refreshed), then check:
```powershell
python -c "import sys,struct; print(sys.version.split()[0], str(struct.calcsize('P')*8)+'-bit')"
```
It must print `3.12.x 64-bit` (or 3.11 / 3.13).

**Step 2 — get the code** (use a short path such as `C:\` — Windows limits paths to 260 characters and the virtual environment is deeply nested)

```powershell
cd C:\
git clone https://github.com/ajx1tech/bypeel.git
cd bypeel
dir      # you should see: app  data  models  sample_data  static  run.bat  requirements.lock.txt ...
```

**Step 3 — run it**

```powershell
.\run.bat          # in PowerShell the  .\  prefix is mandatory
```
(In Command Prompt: `run.bat`.) Expected output matches the Linux example above. The script never
needs administrator rights and never needs `Set-ExecutionPolicy`, because it calls
`.venv\Scripts\python.exe` directly instead of activating the environment.

**Step 4 — open <http://127.0.0.1:8000>** in a browser, then continue with §8.5.

<details>
<summary><b>Prefer typing the commands yourself?</b> (exactly what <code>run.bat</code> does)</summary>

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.lock.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
`py -3.12` is the Python Launcher that the python.org installer provides; use `python` if you do not have it.
</details>

### 8.5 Verification checklist (what to click)

1. **Health** — open <http://127.0.0.1:8000/api/health>. Expect `"status":"ok"`, 17 `model_features`, and `"shap_available":true`.
2. **Dashboard** — <http://127.0.0.1:8000> opens on **1 · Ingest**.
3. **Zero external requests** — press **F12 → Network → reload**. Every row (`d3.min.js`, `chart.umd.min.js`, `papaparse.min.js`, …) is served from `127.0.0.1:8000` with status 200. Nothing leaves the machine.
4. **Run the pipeline** — click **Load Demo Dataset & Analyze**, *or* upload `sample_data/labeled_transactions.csv`, keep **Use Python ML backend** ticked, and click **Process Dataset**. The six pipeline stages complete in seconds and the Overview, Graph Explorer, and Alerts tabs fill with results.
5. **Local persistence** — return to **1 · Ingest → Saved Cases** and click **Open** on your run. It reloads instantly from the local SQLite file `data/bypeel.db` without reprocessing.

### 8.6 Stop, restart, update, reset

| Task | Command |
|---|---|
| Stop the server | `Ctrl+C` in its terminal |
| Start again later | Run `./run.sh` / `.\run.bat` again — it skips everything already installed and starts in seconds |
| Use another port | Linux: `PORT=8001 ./run.sh` · PowerShell: `$env:PORT=8001; .\run.bat` · CMD: `set "PORT=8001" & run.bat` |
| Get the latest code | `git pull`, then run the script again |
| Clean reinstall | Delete the `.venv` folder (`rm -rf .venv` / `Remove-Item -Recurse -Force .venv`) and run the script again |
| Wipe all saved cases | Delete `data/bypeel.db` (recreated automatically) |

### 8.7 Pinned dependency versions

`requirements.txt` lists the direct dependencies and `requirements.lock.txt` pins **every**
package, including transitive ones, with environment markers for Windows / Linux / macOS and
Python 3.11–3.13. The scripts install from the lockfile, so a judge's install is the same on
every machine and cannot be broken by a package that was released the day before.

| Package | Version | Why it is pinned |
|---|---|---|
| Python | 3.11 – 3.13 (64-bit) | scikit-learn 1.9.x needs ≥ 3.11; shap/numba stack needs ≤ 3.13 |
| scikit-learn | **1.9.0** | exact version that pickled the bundled model → loads with zero warnings |
| numpy / pandas | 2.2.6 / 2.2.3 | have wheels for 3.11–3.13 on Windows, Linux, macOS |
| shap · numba · llvmlite | 0.48.0 · 0.61.2 · 0.44.0 | the SHAP explainer; 0.48.0 ships Python 3.13 wheels (0.47.2 does not) |
| networkx · geoip2 · joblib | 3.4.2 · 4.8.0 · 1.4.2 | graph engine · offline GeoIP reader · model loading |
| fastapi · uvicorn · python-multipart | 0.115.6 · 0.32.1 · 0.20 | local API server and file upload |
| reportlab | 4.2.5 | optional server-side PDF export |

<details>
<summary><b>Verification matrix — what has actually been tested</b></summary>

| Check | Result |
|---|---|
| `pip install` of the lockfile, wheels only, Linux x86-64, Python **3.11 / 3.12 / 3.13**; `pip check` clean | ✅ |
| On each of those: bundled model loads with **0 warnings**, SHAP `TreeExplainer`, GeoIP lookup, NetworkX graph | ✅ (warnings treated as errors) |
| `run.sh` from an empty directory: auto-selects Python 3.12 although `python3` was 3.10 → venv → install → self-check → server → `/api/health` | ✅ |
| `run.sh` refuses to start on Python 3.10 only, with an install hint | ✅ |
| Fully offline install from a `wheelhouse/` (56 wheels, 143 MB, `--no-index`) and server start | ✅ |
| Wheel availability of every pin for **Windows x64** and **macOS**, Python 3.11–3.13 (`--only-binary` resolution) | ✅ |
| `run.bat` | Same logic as `run.sh`; **not yet executed on a physical Windows PC** — if anything misbehaves, the four manual PowerShell commands in §8.4 are equivalent |
</details>

---

## 9. Using It

1. Go to **"1 · Ingest"**.
2. Upload your own CSV/JSON/XML under **"Ingest Your Own Metadata"** (or
   click **"Load Demo Dataset"** for the bundled synthetic corpus).
3. **"Use Python ML backend" is checked by default** — click **"Process
   Dataset"**. This uploads your file(s) to `/api/process`, runs the full
   Python pipeline (GeoIP → features → Union-Find → IsolationForest → SHAP
   → motif mining → NetworkX graph), **saves the result as a case in
   `data/bypeel.db`**, and renders it across Overview / Graph Explorer /
   Alerts.
4. Come back later and reopen any past run instantly from the **"Saved
   Cases"** panel at the bottom of the Ingest tab — no re-processing needed.
5. In **Graph Explorer**, toggle between the node-link canvas and the
   **Entity Clusters** view to see wallets resolved into logical entities
   via Union-Find/CIOH.
6. In the **Dossier** tab, select alerts, compute SHA-256 verification
   seals, and export a PDF/JSON intelligence brief — also saved to the
   local database automatically.

---

## 10. Dataset and Schema

Since real seized/intercepted Bitcoin data is sensitive and can't be
distributed, this project — like the problem statement itself specifies —
uses a **synthetic dataset** modeled on real Bitcoin P2P/transaction fields.

**Minimum schema** (all consumed by `app/ingest.py`'s canonical column
mapping, case-insensitive with common aliases):

| Field | Description |
|---|---|
| `timestamp` | Transaction broadcast time |
| `src_ip`, `dst_ip` | Network-layer relay endpoints |
| `txid` | Transaction ID |
| `input_addresses[]`, `output_addresses[]` | Wallets involved |
| `input_amounts[]`, `output_amounts[]` | BTC amounts |
| `fee`, `script_type` | Transaction metadata |
| `geo_country`, `asn` | Derived automatically via offline GeoIP if not already present |

`sample_data/` ships a synthetic corpus built to this exact schema (10,126
transactions with 5 injected laundering motif classes) for immediate
testing — see `sample_data/labeled_transactions.csv`.

---

## 11. Methodology

**Approach.** Transactions are ingested and normalized, enriched with
offline GeoIP/ASN data, and reduced to a 17-dimensional feature vector
capturing structural shape (input/output counts, amount ratios), timing
(hour-of-day, off-hours flag, IP burst frequency), and network context (ASN
hosting risk, IP/wallet diversity, country rarity). Wallets are
simultaneously resolved into logical entities via Union-Find over the
Common-Input-Ownership Heuristic, and every transaction/wallet/IP becomes a
node in a NetworkX heterogeneous graph.

**Model choice.** `IsolationForest` was chosen because illicit Bitcoin
activity is a tiny, unlabeled-in-the-wild minority class — tree-based
isolation naturally isolates outliers in high-dimensional space without
requiring balanced or even labeled training data, and it composes well with
exact SHAP explanations. A complementary rule-based motif layer
(`app/motifs.py`) encodes known laundering structures (peeling chains,
smurfing, layering) using population-relative percentile thresholds
computed fresh from whatever corpus is ingested, so detection generalizes
rather than overfitting to one synthetic generator's quirks.

**Composite scoring.** Each transaction's final risk score blends the
ML anomaly score with a motif-membership bonus (`app/pipeline.py`), giving
a single ranked, filterable 0–1 score per alert with a CRITICAL/HIGH/
MEDIUM/LOW severity band.

---

## 12. Explainability

Every flagged transaction carries:

- A numeric **risk score** and severity band
- The **top contributing features**, via exact SHAP `TreeExplainer`
  attribution (falls back to a robust median/MAD z-score decomposition if
  `shap` isn't installed, so the UI never breaks)
- A plain-language **motif description**
- A reconstructed **evidentiary subgraph** for the flagged transaction

**Example alert, as bypeel actually renders it:**
> Flagged **CRITICAL** (score 0.91) — **Peeling Chain**. Top contributing
> features: `total_input_amount` far above the 99.5th percentile of the
> ingested corpus (z = 4.8), 1-input/2-output shape with strong output
> asymmetry, off-hours broadcast timing. Traced forward through 3 hops of
> peeled outputs before the trail terminates.

This satisfies "why was this flagged" with a genuine glass-box method, not
a black-box score with a decorative bar chart bolted on.

---

## 13. The Local Database

`data/bypeel.db` is a single SQLite file (Python's built-in `sqlite3`
module — zero extra dependency, zero configuration, zero network). Two
tables:

- **`cases`** — every dataset processed via `/api/process`: name, upload
  timestamp, source filenames, summary KPIs, and the full result bundle
  (JSON) so it can be reloaded byte-for-byte identical without re-running
  the ML pipeline.
- **`dossiers`** — every compiled intelligence brief exported from the
  Dossier tab, linked back to its case, with investigator name, notes, and
  the SHA-256 chain-of-custody hash.

Back up or move your entire case history by copying this one file. Delete
it to reset the platform to a blank slate (it's recreated automatically on
next server start).

---

## 14. Problem-Statement Requirement Mapping

### Background
Addressed by fusing **network-layer** telemetry (`src_ip`/`dst_ip`, ASN,
country, burst timing) with **blockchain-layer** data (wallet addresses,
TXIDs, amounts) into one graph — see `app/graph_engine.py` and the Graph
Explorer tab.

### Challenge Objectives
| Objective | Where |
|---|---|
| Ingest & parse bulk metadata (timestamp, src/dst IP & port, TXID, in/out wallets, amounts, fee, script type) | `app/ingest.py` — CSV/JSON/XML, case-insensitive column mapping |
| Build an entity/transaction graph linking IPs, wallets, transactions | `app/graph_engine.py` — NetworkX `MultiDiGraph`, node kinds `ip`/`wallet`/`tx` |
| AI/ML detection with a **working model — not just rules** | `app/ml_engine.py` — trained `IsolationForest` persisted via `joblib`, scored per-transaction; `app/motifs.py` supplies interpretable rule-based motif labels on top |
| Ranked, explainable alert list with confidence score | `app/pipeline.py` composite score + `app/ml_engine.py` SHAP `TreeExplainer` per-feature contribution bars, rendered in the Alerts tab |
| Dashboard / link-analysis visualization | `static/bypeel.html` — Overview KPIs & charts, Graph Explorer node-link canvas, Entity Cluster (CIOH) bubble map |

### Suggested AI/ML Focus Areas
| Focus area | Implementation |
|---|---|
| **Entity Clustering** — group wallets likely owned by one entity using common-input-ownership + graph embeddings | `app/graph_engine.py` `resolve_entities()` — Union-Find CIOH; frontend "Entity Clusters" view visualizes every resolved entity as a packed-circle cluster. *(Graph-embedding half — Node2Vec/GraphSAGE — is a named gap; see §19.)* |
| **Anomaly Detection** — flag statistically unusual transactions/flows | `app/ml_engine.py` `IsolationForest.score_samples()` over the 17-feature vector |
| **Peeling-Chain / Mixing Detection** — detect laundering-pattern transaction sequences | `app/motifs.py` (`peel_chain`, `fan_in_smurfing`, `rapid_fan_out_layering` rules) + `app/analytics.py` `trace_chains()` multi-hop peel-chain tracer, visualized with animated fund-flow playback. *(CoinJoin-specific entropy detection is a named gap; see §19.)* |
| **Risk Scoring** — propagate risk scores from seed illicit wallets via algorithms | The composite score combines the ML anomaly score with motif-membership; the "tainted transit wallet" coloring propagates flagged-transaction status one hop through the entity graph. *(Full decaying-PageRank multi-hop propagation is a named gap; see §19.)* |

### Dataset: Parameters & Synthetic Generation
All minimum fields (`timestamp, src_ip, dst_ip, src_port, dst_port, txid,
input_addresses[], output_addresses[], input_amounts[], output_amounts[],
geo_country/asn`) are consumed by `app/ingest.py`'s canonical schema, with
`geoip_lookup.py` deriving `geo_country`/`asn` offline when a dataset
doesn't already include them.

### Expected Solution / Deliverables
| Deliverable | Status |
|---|---|
| Workable complete offline solution for Linux platform | ✅ `run.sh` — see §8. `run.bat` additionally covers Windows. |
| Working prototype (code repo) with ingestion, correlation, and AI/ML model | ✅ This repository |
| Short technical write-up: approach, model choice, explainability method | ✅ §11–§12 above |
| Dashboard/visualization showing flagged entities and evidence for each flag | ✅ Alerts tab's Intelligence Card shows per-alert SHAP feature bars, evidentiary subgraph, and composite score breakdown for every flag |

---

## 15. Existing Solutions and Prior Art

| Tool | Approach | Gap bypeel addresses |
|---|---|---|
| **Chainalysis Reactor / Elliptic / TRM Labs** | On-chain wallet clustering + proprietary entity attribution | Closed-source, cloud-hosted, €60K–250K+/year, no network-layer (IP/ASN) correlation |
| **GraphSense / BlockSci** (open-source) | Fast on-chain graph clustering + tagging | Infrastructure/query tools, not investigator-facing products — no built-in ML, no SHAP explainability, no dashboard out of the box |
| **Academic P2P deanonymization research** (e.g. Biryukov et al.) | IP-to-transaction linkage via P2P observation | No integrated ML, clustering, or dashboard — a technique, not a platform |
| **Other public SIH-26146 submissions** | Vary — some validate against the real Elliptic/Elliptic++ dataset with full ML benchmarks and richer taint-propagation/embedding claims; several use Streamlit or a dedicated `dashboard/` module rather than a single HTML file | Where other teams are ahead (public-dataset validation, PageRank-style propagation, graph embeddings), those are tracked as open items in §19 rather than glossed over |

**Our honest positioning:** we're not claiming to out-cluster Chainalysis's
proprietary attribution database, and at least one other public submission
for this exact problem statement has validated against a larger, real
academic benchmark than we currently have. What bypeel does claim, and can
demonstrate live: a genuinely offline, zero-cost, install-and-run system
with a real trained model wired end-to-end to real explainability, running
today — see §19 for exactly what's next to close the gap further.

---

## 16. Retraining the Model

```bash
python scripts/train_model.py path/to/your_transactions.csv models/isolation_forest.pkl
```
Re-derives the 17 engineered features from your CSV, fits a fresh
`StandardScaler` + `IsolationForest`, and overwrites the model bundle
in-place. The backend picks it up automatically on next restart.

---

## 17. Running Fully Air-Gapped

At runtime bypeel needs nothing from the internet: the three frontend libraries are served
from `static/vendor/`, GeoIP reads `data/geoip/GeoLite2-Country.mmdb` from disk, and the model
loads from `models/isolation_forest.pkl`. No `fetch()` in the dashboard targets an external host.
The **only** thing that ever comes from the internet is the Python packages, once, via `pip`.
Choose how to get them onto the machine:

### Option A — one-time online install (simplest)
Follow §8 on a connected machine. After the first successful run you can unplug the network
permanently; `./run.sh` / `.\run.bat` keep working and the self-check confirms the model and
GeoIP database load from local disk.

### Option B — the target machine is never connected (wheelhouse)
Build a folder of pre-downloaded packages on **any connected machine that has the same OS, CPU
architecture and Python minor version (e.g. Windows x64 + 3.12) as the target**, copy it next
to the code, and the launch scripts install from it with `--no-index`.

```bash
# Linux (connected machine, Python 3.12) — run inside the bypeel folder:
python3.12 -m pip download --only-binary=:all: -r requirements.lock.txt -d wheelhouse
```
```powershell
# Windows (connected machine, Python 3.12) — run inside the bypeel folder:
py -3.12 -m pip download --only-binary=:all: -r requirements.lock.txt -d wheelhouse
```
This produces about 56 files (≈ 145 MB). Copy the **whole `bypeel` folder, including `wheelhouse/`**,
to the air-gapped machine (USB drive / internal share) and run `./run.sh` or `.\run.bat`. The script
prints `wheelhouse found - installing 100% offline from local wheels...`.
Python itself must already be installed on the target (download the installer once from python.org).
A wheelhouse built on Linux cannot be used on Windows, or the other way round.

### Prove it to a judge
Disconnect Wi-Fi / unplug Ethernet, start bypeel, and repeat the checklist in §8.5 — everything works.
In the browser's Network tab every request is to `127.0.0.1`.

---

## 18. Troubleshooting

Find your message in the table. If it is not there, jump to **"Collect diagnostics"** below.

| What you see | Cause | Fix |
|---|---|---|
| `no 64-bit Python 3.11, 3.12 or 3.13 found` (printed by `run.sh` / `run.bat`) | Python is missing, too old (3.10 or earlier), too new (3.14), or 32-bit | Install **64-bit Python 3.12** (§8.3 step 1 / §8.4 step 1), open a **new** terminal, run the script again. |
| `ERROR: Could not find a version that satisfies the requirement scikit-learn==1.9.0 … No matching distribution found` | You ran `pip install` by hand on Python 3.10 or older | Use Python 3.11–3.13. scikit-learn 1.9.0 is pinned on purpose to match the bundled model. |
| `Could not find vswhere.exe` / `meson setup … failed` / `error: metadata-generation-failed` / `Microsoft Visual C++ 14.0 or greater is required` | pip tried to **compile** a package because no wheel matched — almost always 32-bit Python or Python 3.14 | Install 64-bit Python 3.12, delete `.venv`, run again. The scripts pass `--only-binary=:all:`, so you should now get a short "No matching distribution" message instead. |
| PowerShell: `run.bat : The term 'run.bat' is not recognized` | PowerShell does not run files from the current folder without a path | Type `.\run.bat` (with `.\`). |
| `'python' is not recognized` or typing `python` opens the Microsoft Store | Python is not on PATH, or the Store "app execution alias" is intercepting it | Reinstall Python and tick **Add python.exe to PATH**; or *Settings → Apps → Advanced app settings → App execution aliases* → turn **off** `python.exe` and `python3.exe`; open a new terminal. `run.bat` also tries `py -3.12`, `py -3.13`, `py -3.11`. |
| `running scripts is disabled on this system` / `PSSecurityException` (only if you try `Activate.ps1` yourself) | PowerShell execution policy | Not needed for bypeel: use the scripts or the manual commands, which call `.venv\Scripts\python.exe` directly. If you still want to activate: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force`. |
| `The virtual environment was not created successfully because ensurepip is not available` (Linux) | Debian/Ubuntu ship `venv` as a separate package | `sudo apt install -y python3-venv` (or `python3.12-venv`), delete `.venv`, run again. |
| `error: externally-managed-environment` | You ran `pip install` outside the virtual environment on a modern distro | Use `./run.sh`, or the manual commands that call `.venv/bin/python -m pip`. |
| `ModuleNotFoundError: No module named 'app'` / `'pandas'` / `'sklearn'` | The server was started with the system Python instead of the `.venv` one, or you are not in the repository root | `cd` into the folder that contains `run.sh`, then start through the script or `.venv/bin/python -m uvicorn …` (`.\.venv\Scripts\python.exe -m uvicorn …` on Windows). |
| `InconsistentVersionWarning` from scikit-learn | A scikit-learn version other than 1.9.0 is installed (the lockfile prevents this) | Delete `.venv` and run the script again; do not run `pip install -U scikit-learn`. |
| `Cannot find path … 'bypeel-repo'` / `can't open file … run.sh` | You are in the wrong folder (a ZIP often unpacks into `bypeel-main/`) | `cd bypeel-main` (or whatever the folder is called); `ls` / `dir` must show `run.sh` and `app`. |
| `[Errno 98] address already in use` / `only one usage of each socket address` | Something already listens on port 8000 (often a previous bypeel) | Use another port: `PORT=8001 ./run.sh`. To find the culprit — Linux: `ss -ltnp \| grep 8000`; Windows: `netstat -ano \| findstr :8000`. |
| pip hangs or says `Connection timed out` / `ProxyError` | Corporate or campus proxy during the one-time install | Set the proxy first — Linux: `export HTTPS_PROXY=http://host:port`; PowerShell: `$env:HTTPS_PROXY="http://host:port"` — then rerun. Or use the offline wheelhouse (§17). |
| Browser shows the old page, or charts are blank | Cached page | Hard refresh: `Ctrl+Shift+R` (`Cmd+Shift+R` on macOS). |
| "Saved Cases" says backend not running | Server is not running, or the page was opened as a file instead of via `http://127.0.0.1:8000` | Start the server and use the URL. |
| `/api/process` returns 500 | Bad input file | The full traceback is in the server terminal. Most common cause: the file has no `timestamp` or `txid` column. |
| GeoIP shows "Unknown" for every IP | Database file missing | Confirm `data/geoip/GeoLite2-Country.mmdb` exists (it ships in the repo). |
| Large dataset (100k+ tx) is slow | `src_ip_burst_count` in `app/features.py` is O(n) per IP group | Pre-aggregate by IP or widen the burst-window bucket for very large corpora. |
| Want real ASN numbers, not just country | Free edition is Country-only | Download MaxMind's separate **GeoLite2-ASN.mmdb** (free account) into `data/geoip/` — auto-detected. |

### Collect diagnostics (paste this output when asking for help)

```bash
# Linux / macOS (inside the bypeel folder)
python3 --version; uname -m; ls
.venv/bin/python --version
.venv/bin/python -m pip list 2>/dev/null | grep -i -E "scikit-learn|numpy|pandas|shap|numba|fastapi|uvicorn"
```
```powershell
# Windows PowerShell (inside the bypeel folder)
python --version; py -0p; $env:PROCESSOR_ARCHITECTURE; dir
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip list | Select-String -Pattern "scikit-learn|numpy|pandas|shap|numba|fastapi|uvicorn"
```

---

## 19. Roadmap

**In progress**
- [ ] Split `static/bypeel.html` into a conventional `static/dashboard/{index.html, css/, js/}` structure for easier navigation, with zero behavior change

**Planned — closing gaps identified against other public submissions**
- [ ] Multi-hop, decaying **Personalized PageRank** taint propagation from seed illicit addresses (`networkx.pagerank` with a personalization vector — extension point already identified in `app/graph_engine.py`)
- [ ] **CoinJoin/mixer detection** via Shannon-entropy analysis of output-value distributions in `app/motifs.py`
- [ ] **Node2Vec/GraphSAGE** graph embeddings clustered with HDBSCAN, layered on top of the existing CIOH Union-Find to catch single-input-wallet entities that never co-spend
- [ ] Validation against a public benchmark dataset (Elliptic/Elliptic++) alongside our own synthetic corpus, for a credibility-comparable precision/recall number
- [ ] `src_port`/`dst_port` added to the canonical ingestion schema (currently accepted but not yet used as model features)

**Completed**
- [x] Multi-format ingestion (CSV/JSON/XML) with schema normalization
- [x] Offline GeoIP/ASN enrichment (MaxMind GeoLite2)
- [x] NetworkX heterogeneous graph + Union-Find CIOH entity resolution
- [x] Trained IsolationForest anomaly detection
- [x] Exact SHAP TreeExplainer explainability
- [x] 5 rule-based laundering motifs with data-driven thresholds
- [x] Multi-hop peel-chain tracer with animated playback
- [x] Composite ranked alert scoring
- [x] Interactive dashboard: Overview, Graph Explorer (node-link + Entity Clusters), Alerts, Dossier & Export
- [x] SHA-256 chain-of-custody sealing + PDF/JSON case export
- [x] Local SQLite case persistence with instant reload
- [x] Client-side JS fallback engine (zero-install demo mode)
- [x] Zero-CDN, fully vendored frontend
- [x] Reproducible install: exact-pinned lockfile, wheels-only, Python 3.11–3.13 on Windows and Linux, with an offline wheelhouse option

---

## 20. Team

**Team Endeavour** — Smart India Hackathon 2026, Problem Statement 26146.

---

*This repository is developed as part of Smart India Hackathon 2026,
Problem Statement 26146. Refer to the sections above for dataset
provenance, usage limitations, deployment requirements, and security
considerations.*
