# bypeel — AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

**A complete, 100% offline, air-gapped system** that ingests bulk Bitcoin
transaction/network metadata (CSV/JSON/XML), correlates network-layer
(IP/port/timing) observations with blockchain-layer (wallet/TXID/amount)
data, applies a trained ML model to detect anomalies and cluster entities,
and generates prioritized, explainable investigative leads through an
interactive dashboard.

Built for **SIH Problem Statement #26146** (NTRO — Blockchain & Cybersecurity).
This document maps every requirement in that problem statement to the exact
file/module that satisfies it — see §9.

**Zero CDN. Zero network calls at runtime.** Every JS library, font, the ML
model, the GeoIP database, and the transaction data itself all live on disk.
The only time this repo touches the network is `pip install` during
first-time setup.

---

## 1. What you get

- A **single-file frontend** (`static/bypeel.html`) — Overview dashboard,
  interactive Graph Explorer (node-link canvas + Entity Cluster/CIOH bubble
  map), ranked Alerts queue with SHAP-style explainability cards, and a
  Dossier & Export tab with SHA-256 chain-of-custody sealing.
- A **Python FastAPI backend** running a real trained **IsolationForest**
  (via `joblib`), **exact SHAP `TreeExplainer`** attributions, a
  **NetworkX** heterogeneous graph, and **Union-Find** entity resolution
  (Common-Input-Ownership Heuristic).
- **Offline GeoIP/ASN enrichment** via a bundled MaxMind GeoLite2 `.mmdb`.
- **Local SQLite persistence** (`data/bypeel.db`) — every dataset you
  process is saved as a "case" on your own machine and can be reopened
  instantly without re-running the pipeline. No cloud, no external DB
  server.
- A **client-side JS fallback engine** so the dashboard also works with
  zero installation (open the HTML file directly) for quick screening demos.

---

## 2. Architecture

```
 ┌───────────────────────────────────────────────────────────────────────┐
 │  Browser — static/bypeel.html  (all libraries vendored locally)         │
 │  static/vendor/d3.min.js · chart.umd.min.js · papaparse.min.js          │
 │                                                                         │
 │  Ingest tab                                                             │
 │    ├─ "Load Demo Dataset"  → bundled 10,126-tx synthetic corpus         │
 │    ├─ "Saved Cases" panel  → GET /api/cases  (reload from SQLite)       │
 │    └─ "Process Dataset"    → POST /api/process (Python backend, default)│
 │                                or in-browser JS engine (no server needed)│
 │                                                                         │
 │  Overview · Graph Explorer · Alerts · Dossier & Export                  │
 └────────────────────────────┬────────────────────────────────────────────┘
                               │ HTTP, localhost only
                               ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │  Backend — app/  (FastAPI, Python 3.10+)                                │
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

**Why two pipelines?** The Python backend is the "real" system — a trained
model (not just z-score rules), exact SHAP attributions, a full NetworkX
graph, and persistent local storage. It is the **default** path (the "Use
Python ML backend" checkbox in the Ingest tab is checked out of the box).
The in-browser JS engine is kept as a zero-install fallback so the frontend
still functions as a standalone artifact if someone opens the HTML file
without ever starting the server. Both produce the identical JSON shape the
dashboard renders.

---

## 3. Repository layout

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
│   ├── bypeel.html               The entire frontend (single file)
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
├── requirements.txt
├── run.sh                    One-command setup + launch (Linux/macOS)
├── run.bat                   One-command setup + launch (Windows)
├── .gitignore
└── README.md
```

**What to push to GitHub:** everything above, as-is. `models/isolation_forest.pkl`
(~2.5 MB), `data/geoip/GeoLite2-Country.mmdb` (~8.6 MB), and the vendored JS
(~500 KB total) are all small enough for a normal `git push` — no Git LFS
needed (GitHub's soft limit is 50 MB/file). `data/bypeel.db` is git-ignored
by design: it's per-installation local case storage, not source code.

---

## 4. Quick start

### Linux / macOS
```bash
git clone <your-repo-url> bypeel
cd bypeel
chmod +x run.sh
./run.sh
```
Open **http://127.0.0.1:8000**.

### Windows
```cmd
git clone <your-repo-url> bypeel
cd bypeel
run.bat
```
Open **http://127.0.0.1:8000**.

Both scripts: create a virtual environment → install dependencies (the only
step that needs internet — pip fetching packages) → sanity-check the model
and GeoIP database load offline → start the server. From then on the
running application makes **zero** network calls.

### Using it
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
5. In the Dossier tab, select alerts, compute SHA-256 verification seals,
   and export a PDF/JSON intelligence brief — this is also saved to the
   local database automatically.

---

## 5. Manual setup (without run.sh/run.bat)

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Requirements: Python 3.10–3.12. No GPU, no external database server, no
internet after the initial `pip install`.

---

## 6. Retraining the model on your own data

```bash
python scripts/train_model.py path/to/your_transactions.csv models/isolation_forest.pkl
```
Re-derives the 17 engineered features from your CSV, fits a fresh
`StandardScaler` + `IsolationForest`, and overwrites the model bundle
in-place. The backend picks it up automatically on next restart.

---

## 7. The local database

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

## 8. Running fully air-gapped (verifying zero network)

Everything is already vendored — there is nothing to download for runtime.
To prove it to a judge on an actual air-gapped machine:
```bash
# disconnect network, then:
./run.sh    # first run must happen WITH internet (pip install); subsequent runs don't need it
```
All three frontend libraries are served from `static/vendor/` by the FastAPI
static mount (`/static/vendor/...`), GeoIP resolution reads
`data/geoip/GeoLite2-Country.mmdb` directly off disk, and the ML model loads
from `models/isolation_forest.pkl` via `joblib`. No `fetch()` in the
frontend ever targets an external host.

---

## 9. Problem-statement requirement mapping

### Background
> *Bitcoin's pseudonymous, peer-to-peer design lets criminal actors move,
> layer, and cash out illicit funds while evading traditional financial
> surveillance.*

Addressed by fusing **network-layer** telemetry (`src_ip`/`dst_ip`, ASN,
country, burst timing) with **blockchain-layer** data (wallet addresses,
TXIDs, amounts) into one graph — see `app/graph_engine.py` and the Graph
Explorer tab.

### Challenge Objectives
| Objective | Where |
|---|---|
| Ingest & parse bulk metadata (timestamp, src/dst IP & port, TXID, in/out wallets, amounts, fee, script type) | `app/ingest.py` — CSV/JSON/XML, case-insensitive column mapping |
| Build an entity/transaction graph linking IPs, wallets, transactions | `app/graph_engine.py` — NetworkX `MultiDiGraph`, node kinds `ip`/`wallet`/`tx` |
| AI/ML detection with a **working model — not just rules** | `app/ml_engine.py` — trained `IsolationForest` persisted via `joblib`, scored per-transaction; `app/motifs.py` supplies interpretable rule-based motif labels on top, exactly matching the "hybrid heuristic + ML" design proposed in the technical approach |
| Ranked, explainable alert list with confidence score | `app/pipeline.py` composite score (`0.65·ML + 0.35·motif bonus`) + `app/ml_engine.py` SHAP `TreeExplainer` per-feature contribution bars, rendered in the Alerts tab |
| Dashboard / link-analysis visualization | `static/bypeel.html` — Overview KPIs & charts, Graph Explorer node-link canvas, Entity Cluster (CIOH) bubble map |

### Suggested AI/ML Focus Areas
| Focus area | Implementation |
|---|---|
| **Entity Clustering** — group wallets likely owned by one entity using common-input-ownership + graph embeddings | `app/graph_engine.py` `resolve_entities()` — Union-Find CIOH; frontend "Entity Clusters" view (`renderClusters()` in Graph Explorer) visualizes every resolved entity as a packed-circle cluster |
| **Anomaly Detection** — flag statistically unusual transactions/flows | `app/ml_engine.py` `IsolationForest.score_samples()` over the 17-feature vector |
| **Peeling-Chain / Mixing Detection** — detect laundering-pattern transaction sequences | `app/motifs.py` (`peel_chain`, `fan_in_smurfing`, `rapid_fan_out_layering` rules) + `app/analytics.py` `trace_chains()` multi-hop peel-chain tracer, visualized with animated fund-flow playback in Graph Explorer |
| **Risk Scoring** — propagate risk scores from seed illicit wallets via algorithms | The composite score in `app/pipeline.py` combines the ML anomaly score with motif-membership; the "tainted transit wallet" coloring in the graph propagates flagged-transaction status one hop through the entity graph (extendable to full decaying-PageRank taint propagation — see §11) |

### Dataset: Parameters & Synthetic Generation
Minimum fields required — `timestamp, src_ip, dst_ip, src_port, dst_port,
txid, input_addresses[], output_addresses[], input_amounts[],
output_amounts[], geo_country/asn` — are all consumed by `app/ingest.py`'s
canonical schema (`CANONICAL_ALIASES`), with `geoip_lookup.py` deriving
`geo_country`/`asn` offline via the bundled GeoLite2 database when a dataset
doesn't already include them. `sample_data/` ships a synthetic corpus
modelled on exactly these fields (10,126 transactions, 5 injected laundering
motif classes) for immediate testing.

### Expected Solution / Deliverables
| Deliverable | Status |
|---|---|
| Workable complete offline solution for Linux platform | ✅ `run.sh` (Linux/macOS) — see §4. `run.bat` additionally covers Windows. |
| Working prototype (code repo) with ingestion, correlation, and AI/ML model | ✅ This repository — `app/` package, `models/isolation_forest.pkl` |
| Short technical write-up: approach, model choice, explainability method | See §10 below |
| Dashboard/visualization showing flagged entities and evidence for each flag | ✅ `static/bypeel.html` — Alerts tab's Intelligence Card shows per-alert SHAP feature bars, evidentiary subgraph, and composite score breakdown for every flag |

---

## 10. Technical approach summary (short write-up)

**Approach.** Transactions are ingested and normalized (see dataset schema
above), enriched with offline GeoIP/ASN data, and reduced to a
17-dimensional feature vector capturing structural shape (input/output
counts, amount ratios), timing (hour-of-day, off-hours flag, IP burst
frequency), and network context (ASN hosting risk, IP/wallet diversity,
country rarity). Wallets are simultaneously resolved into logical entities
via Union-Find over the Common-Input-Ownership Heuristic, and every
transaction/wallet/IP becomes a node in a NetworkX heterogeneous graph.

**Model choice.** `IsolationForest` (`scikit-learn`) was chosen because
illicit Bitcoin activity is a tiny, unlabeled-in-the-wild minority class —
tree-based isolation naturally isolates outliers in high-dimensional space
without requiring balanced or even labeled training data, and it composes
well with exact SHAP explanations (unlike, say, a black-box deep model). A
complementary rule-based motif layer (`app/motifs.py`) encodes known
laundering structures (peeling chains, smurfing, layering) using
population-relative percentile thresholds computed fresh from whatever
corpus is ingested — so detection generalizes rather than overfitting to
one synthetic generator's quirks.

**Explainability method.** Every alert carries an exact SHAP
`TreeExplainer` decomposition (falls back to a robust median/MAD z-score
attribution if `shap` isn't installed) showing the top contributing
features and their direction, plus a plain-language motif description and a
reconstructed evidentiary subgraph — satisfying the "why was this flagged"
requirement with a genuine glass-box method rather than a black box.

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` on startup | Activate the venv first, or re-run `pip install -r requirements.txt`. |
| `InconsistentVersionWarning` from sklearn | Harmless — the bundled model was pickled with a slightly different scikit-learn point release. Safe to ignore, or retrain with §6 to silence it. |
| `/api/process` returns 500 | Check the terminal running uvicorn — the full traceback is logged there. Most common cause: uploaded file is missing a `timestamp` or `txid` column. |
| GeoIP shows "Unknown" for every IP | Confirm `data/geoip/GeoLite2-Country.mmdb` exists (it ships in this repo — don't delete it). |
| Port 8000 already in use | `uvicorn app.main:app --host 127.0.0.1 --port 8001` and open that port instead. |
| "Saved Cases" panel says backend not running | Start the server with `./run.sh` / `run.bat` — the client-side JS engine works standalone but doesn't persist to SQLite. |
| Large dataset (100k+ tx) is slow | `src_ip_burst_count` in `app/features.py` is O(n) per IP group; for very large corpora, pre-aggregate by IP or widen the burst window bucket size. |
| Want real ASN numbers, not just country | Download the separate (free, MaxMind-account-gated) **GeoLite2-ASN.mmdb** and drop it in `data/geoip/` — `app/geoip_lookup.py` auto-detects and uses it. |

---

## 12. Future extensions (not required by the problem statement, noted for completeness)

- Full **Decaying Personalized PageRank** taint propagation from seed
  illicit addresses (the current graph coloring does 1-hop taint marking;
  `networkx.pagerank` with a personalization vector is a drop-in extension
  point in `app/graph_engine.py`).
- **GraphSAGE / Node2Vec** structural embeddings clustered with HDBSCAN,
  layered on top of the existing CIOH Union-Find for single-input-wallet
  entity expansion (mentioned as an "X-factor" stretch goal in the original
  proposal).
- **ReportLab**-generated formal PDF intelligence briefs (the Dossier tab
  currently uses the browser's native print-to-PDF, which is fully offline
  and already produces a court-formatted document — `reportlab` is listed in
  `requirements.txt` for teams that want a server-rendered PDF instead).
