"""Peel-chain hop tracing and Overview-tab summary statistics."""
from collections import defaultdict
import pandas as pd
import numpy as np


def trace_chains(df, max_chains=12, max_hops=5):
    addr_in_idx = defaultdict(list)
    for idx, ins in zip(df.index, df["input_list"]):
        for a in ins:
            addr_in_idx[a].append(idx)

    peel_rows = df[df["_motif"] == "peel_chain"]
    chains = []
    for _, row in peel_rows.iterrows():
        outs = sorted(zip(row["output_list"], row["output_amt_list"]), key=lambda x: -x[1])
        if len(outs) < 2:
            continue
        big_addr, big_amt = outs[0]
        small_addr, small_amt = outs[1]
        hops = [{"hop": 0, "txid": row["txid"], "timestamp": str(row["timestamp"]),
                 "peeled_off": {"addr": small_addr, "amount": round(small_amt, 6)},
                 "change_forward": {"addr": big_addr, "amount": round(big_amt, 6)},
                 "src_ip": row["src_ip"], "dst_ip": row["dst_ip"]}]
        cur_addr = big_addr
        seen_tx = {row["txid"]}
        for h in range(1, max_hops + 1):
            cand = [j for j in addr_in_idx.get(cur_addr, [])
                    if df.loc[j, "txid"] not in seen_tx and df.loc[j, "_ts"] > pd.Timestamp(hops[-1]["timestamp"], tz="UTC")]
            if not cand:
                break
            j = cand[0]
            r2 = df.loc[j]
            seen_tx.add(r2["txid"])
            outs2 = sorted(zip(r2["output_list"], r2["output_amt_list"]), key=lambda x: -x[1])
            if not outs2:
                break
            nxt_addr, nxt_amt = outs2[0]
            peeled2 = outs2[1] if len(outs2) > 1 else None
            hops.append({"hop": h, "txid": r2["txid"], "timestamp": str(r2["timestamp"]),
                         "peeled_off": {"addr": peeled2[0], "amount": round(peeled2[1], 6)} if peeled2 else None,
                         "change_forward": {"addr": nxt_addr, "amount": round(nxt_amt, 6)},
                         "src_ip": r2["src_ip"], "dst_ip": r2["dst_ip"]})
            cur_addr = nxt_addr
        if len(hops) >= 2:
            chains.append({"seed_txid": row["txid"], "seed_country": row.get("src_country", "Unknown"),
                           "seed_asn": int(row["src_asn"]) if pd.notna(row.get("src_asn")) else 0,
                           "total_hops": len(hops) - 1, "hops": hops})

    chains.sort(key=lambda c: -c["total_hops"])
    return chains[:max_chains]


def build_overview(df, alerts, entity_members):
    anomaly_counts = df[df["_motif"] != "none"]["_motif"].value_counts().to_dict()
    script_counts = df["script_type"].value_counts().to_dict()
    country_counts = df["src_country"].value_counts().head(12).to_dict()

    df = df.copy()
    df["_date"] = df["_ts"].dt.date.astype(str)
    timeline = df.groupby("_date").size().to_dict()
    anom_by_date = df[df["_motif"] != "none"].groupby("_date").size().to_dict()

    scores = [a["score"] for a in alerts]
    if scores:
        lo, hi = min(scores), max(scores)
        bins = 10
        w = (hi - lo) / bins or 1
        hist = [0] * bins
        for s in scores:
            idx = min(bins - 1, max(0, int((s - lo) / w)))
            hist[idx] += 1
        score_hist = [{"bin": f"{lo + i * w:.2f}-{lo + (i + 1) * w:.2f}", "count": c} for i, c in enumerate(hist)]
    else:
        score_hist = []

    asn_counts = {}
    for a in alerts:
        org = a.get("src_asn_org")
        if org and org != "Unknown":
            asn_counts[org] = asn_counts.get(org, 0) + 1
    asn_counts = dict(sorted(asn_counts.items(), key=lambda x: -x[1])[:8])

    has_labels = df["is_anomaly"].notna().any() if "is_anomaly" in df.columns else False
    top50_precision = top100_precision = true_positive_alerts = None
    if has_labels:
        by_txid = df.set_index("txid")["is_anomaly"].to_dict()
        top50 = alerts[:50]
        top100 = alerts[:100]
        tp = lambda arr: sum(1 for a in arr if by_txid.get(a["txid"]) == 1)
        top50_precision = round(tp(top50) / len(top50) * 100, 1) if top50 else None
        top100_precision = round(tp(top100) / len(top100) * 100, 1) if top100 else None
        true_positive_alerts = sum(1 for a in alerts if a.get("is_anomaly") == 1)

    multi_entities = sum(1 for members in entity_members.values() if len(members) > 1)

    return {
        "kpis": {
            "total_tx": int(len(df)),
            "total_wallets": int(sum(len(v) for v in entity_members.values())),
            "total_entities": int(len(entity_members)),
            "deanon_clusters": int(multi_entities),
            "total_alerts": int(len(alerts)),
            "true_positive_alerts": true_positive_alerts,
            "top50_precision": top50_precision,
            "top100_precision": top100_precision,
            "flagged_tx": int((df["_motif"] != "none").sum()),
            "countries": int(df["src_country"].nunique()),
            "has_ground_truth": bool(has_labels),
        },
        "anomaly_counts": anomaly_counts,
        "script_counts": script_counts,
        "country_counts": country_counts,
        "timeline": timeline,
        "anom_by_date": anom_by_date,
        "score_hist": score_hist,
        "asn_org_counts": asn_counts,
    }
