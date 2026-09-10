"""
bypeel — offline FastAPI backend.

Serves the static single-file frontend (static/bypeel.html) and exposes a
local REST API that runs the full Python ML/graph pipeline (GeoIP ->
features -> Union-Find CIOH -> IsolationForest/SHAP -> motif mining ->
NetworkX graph) over uploaded transaction files, persists every processed
result as a "case" in a local SQLite database (data/bypeel.db), and returns
the exact JSON bundle the frontend renders.

Run:
    uvicorn app.main:app --host 127.0.0.1 --port 8000
Then open http://127.0.0.1:8000 in a browser. Zero network calls anywhere in
this process or the served frontend — GeoIP, the ML model, the JS
libraries (D3/Chart.js/PapaParse), and the database are all local files.
"""
import logging
import traceback

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .ingest import load_many
from .pipeline import run_pipeline, get_ml_engine
from . import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("bypeel")

app = FastAPI(title="bypeel API", description="Offline Bitcoin transaction intelligence backend")
db.init_db()


@app.get("/")
def serve_frontend():
    return FileResponse("static/bypeel.html")


@app.get("/api/health")
def health():
    """Confirms the ML model loads correctly (useful as a first setup sanity check)."""
    try:
        engine = get_ml_engine()
        return {"status": "ok", "model_features": engine.feature_names,
                "shap_available": engine._explainer is not None}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@app.post("/api/process")
async def process(
    tx_files: list[UploadFile] = File(..., description="One or more transaction metadata files (CSV/JSON/XML)"),
    case_name: str = Form(default=None),
):
    """
    Runs the complete offline pipeline over the uploaded transaction file(s),
    PERSISTS the result to the local SQLite database as a new case, and
    returns the bundle {overview, alerts, graph, chains, case_id} the
    frontend renders directly via window.BYPEEL_applyDataset(...).
    """
    try:
        files = []
        for f in tx_files:
            content = await f.read()
            files.append((f.filename, content))

        df = load_many(files)
        if df.empty:
            return JSONResponse(status_code=400, content={"error": "No valid transaction rows parsed from the uploaded file(s)."})

        logger.info("Processing %d transactions from %d file(s)", len(df), len(files))
        result = run_pipeline(df, log=logger.info)

        name = case_name or f"Case — {', '.join(f[0] for f in files)}"
        case_id = db.save_case(name, [f[0] for f in files], result)
        result["case_id"] = case_id
        logger.info("Saved as case %s", case_id)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error("Pipeline error: %s\n%s", e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/cases")
def api_list_cases():
    """List every previously processed case saved in the local database."""
    return db.list_cases()


@app.get("/api/cases/{case_id}")
def api_get_case(case_id: str):
    """Reload a previously processed case's full result bundle instantly (no re-processing)."""
    case = db.get_case(case_id)
    if not case:
        return JSONResponse(status_code=404, content={"error": "Case not found"})
    bundle = case["bundle"]
    bundle["case_id"] = case_id
    return JSONResponse(content=bundle)


@app.delete("/api/cases/{case_id}")
def api_delete_case(case_id: str):
    ok = db.delete_case(case_id)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "Case not found"})
    return {"status": "deleted", "case_id": case_id}


@app.post("/api/dossiers")
async def api_save_dossier(payload: dict):
    """Persist a compiled intelligence dossier (selected alerts + hashes) to the local DB."""
    case_id = payload.get("case_id", "unsaved")
    dossier_id = db.save_dossier(
        case_id=case_id,
        investigator=payload.get("investigator", "Unattributed"),
        notes=payload.get("notes", ""),
        report_hash=payload.get("report_hash_sha256"),
        dossier_payload=payload,
    )
    return {"status": "saved", "dossier_id": dossier_id}


@app.get("/api/dossiers")
def api_list_dossiers(case_id: str = None):
    return db.list_dossiers(case_id)


# static assets last so /api/* and / above take precedence
app.mount("/static", StaticFiles(directory="static"), name="static")

