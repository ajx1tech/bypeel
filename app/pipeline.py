"""
End-to-end offline pipeline, orchestrating every module in this package into
the exact JSON bundle shape the bypeel.html frontend consumes:

    { overview: {...}, alerts: [...], graph: {nodes, edges}, chains: [...] }

Stages (mirrors the 6-stage pipeline shown in the UI):
  1. Ingestion            -> app.ingest
  2. Geo/ASN enrichment   -> app.geoip_lookup
  3. Feature engineering  -> app.features
  4. Entity resolution    -> app.graph_engine (Union-Find / CIOH)
  5. ML scoring + SHAP    -> app.ml_engine (IsolationForest, joblib-persisted)
  6. Motif mining + graph -> app.motifs, app.graph_engine, app.analytics
"""
import time
import pandas as pd

from . import geoip_lookup, features as feat_mod, graph_engine, motifs, analytics
from .ml_engine import MLEngine

_ml_engine_singleton = None


def get_ml_engine():
    global _ml_engine_singleton
    if _ml_engine_singleton is None:
        _ml_engine_singleton = MLEngine()
    return _ml_engine_singleton


def run_pipeline(raw_df: pd.DataFrame, log=print, max_alerts=300):
    t0 = time.time()
    df = raw_df.copy()

    log(f"[1/6] Ingested {len(df):,} transaction records.")

    log("[2/6] Enriching source/destination IPs via offline MaxMind GeoLite2…")
    df = geoip_lookup.enrich_dataframe(df)

    log("[3/6] Computing 17-feature vector for anomaly scoring…")
    df = feat_mod.build_features(df)

    log("[4/6] Resolving wallet entities via Common-Input-Ownership Heuristic (Union-Find)…")
    uf, eid_fn, entity_members = graph_engine.resolve_entities(df)
    multi = sum(1 for v in entity_members.values() if len(v) > 1)
    log(f"       Resolved {len(uf.parent):,} wallets into {len(entity_members):,} logical entities "
        f"({multi} multi-wallet / de-anonymized).")

    log("[5/6] Scoring transactions with IsolationForest + generating SHAP attributions…")
    engine = get_ml_engine()
    X_df = feat_mod.feature_matrix(df)
    raw_scores, norm_scores, Xs = engine.score(X_df)
    df["_ml_score"] = norm_scores
    shap_feats = engine.explain(Xs, top_n=4)

    log("[6/6] Mining laundering motifs and building the knowledge graph…")
    df, thresholds = motifs.detect_motifs(df)
    log(f"       Motif thresholds -> {thresholds}")

    # composite score: blend ML anomaly score with a motif bonus (mirrors the
    # Composite Risk Score formula in the problem statement)
    motif_bonus = df["_motif"].apply(lambda m: 0.35 if m != "none" else 0.0)
    df["_score"] = (df["_ml_score"] * 0.65 + motif_bonus).clip(0, 0.99)

    ranked = df.sort_values("_score", ascending=False).reset_index(drop=True)
    ranked_feats = [shap_feats[i] for i in df.sort_values("_score", ascending=False).index]

    alerts = []
    for i, row in ranked.head(max_alerts).iterrows():
        score = float(row["_score"])
        severity = "CRITICAL" if score >= 0.6 else "HIGH" if score >= 0.4 else "MEDIUM" if score >= 0.25 else "LOW"
        is_anom = int(row["is_anomaly"]) if pd.notna(row.get("is_anomaly")) else (1 if row["_motif"] != "none" else 0)
        alerts.append({
            "rank": i + 1, "score": round(score, 4), "severity": severity,
            "txid": row["txid"], "timestamp": str(row["timestamp"]),
            "src_ip": row["src_ip"], "dst_ip": row["dst_ip"],
            "src_country": row.get("src_country", "Unknown"),
            "src_asn": int(row["src_asn"]) if pd.notna(row.get("src_asn")) else None,
            "src_asn_org": row.get("src_asn_org", "Unknown"),
            "num_inputs": int(row["num_inputs"]), "num_outputs": int(row["num_outputs"]),
            "total_input": round(float(row["total_input_amount"]), 6),
            "total_output": round(float(row["total_output_amount"]), 6),
            "fee": round(float(row["fee"]), 8),
            "ip_wallet_diversity": int(row["ip_wallet_diversity"]),
            "is_anomaly": is_anom, "type": row["_motif"],
            "type_label": motifs.TYPE_LABELS[row["_motif"]], "type_desc": motifs.TYPE_DESC[row["_motif"]],
            "features": ranked_feats[i],
        })

    G = graph_engine.build_networkx_graph(df, eid_fn)
    graph_json = graph_engine.graph_to_frontend_json(G, df, eid_fn)
    chains = analytics.trace_chains(df)
    overview = analytics.build_overview(df, alerts, entity_members)

    elapsed = round(time.time() - t0, 2)
    log(f"Pipeline complete in {elapsed}s. NetworkX graph: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges. {len(alerts)} alerts ranked.")

    return {"overview": overview, "alerts": alerts, "graph": graph_json, "chains": chains}
