"""
Laundering-motif mining.

Detects the five behavioural motifs called out in the problem statement:
peeling chains, CoinJoin-style fan-in smurfing, rapid fan-out layering,
round-amount off-hours transfers, and shared-IP bursts.

Thresholds are derived from the ingested population's own percentiles/robust
z-scores rather than hard-coded constants, so the detector generalizes to
whatever corpus is uploaded (mirrors the validated client-side JS detector
shipped in the frontend's Ingest tab).
"""
import numpy as np


def _quantile(s, q):
    return float(np.quantile(s, q)) if len(s) else 0.0


def _mad(s, med):
    dev = np.abs(s - med)
    m = np.median(dev)
    if m > 0:
        return m
    mean_dev = dev.mean() if len(dev) else 0
    return mean_dev if mean_dev > 0 else max(abs(med) * 0.1, 0.5)


def detect_motifs(df):
    df = df.copy()
    outs_sorted = df["output_amt_list"].apply(lambda l: sorted(l, reverse=True))
    df["_out_asym"] = outs_sorted.apply(lambda l: (l[0] / l[-1]) if len(l) >= 2 and l[-1] > 0 else (1.0 if len(l) == 1 else 0.0))

    med_nout, mad_nout = df["num_outputs"].median(), _mad(df["num_outputs"].values, df["num_outputs"].median())
    med_nin, mad_nin = df["num_inputs"].median(), _mad(df["num_inputs"].values, df["num_inputs"].median())
    fan_out_thresh = max(8, med_nout + 6 * max(1, mad_nout))
    fan_in_thresh = max(8, med_nin + 6 * max(1, mad_nin))
    peel_value_q995 = _quantile(df["total_input_amount"].values, 0.995)
    burst_q995 = _quantile(df["src_ip_burst_count"].values, 0.995)
    ip_burst_thresh = max(4, int(np.ceil(burst_q995)))

    def classify(row):
        if row["num_inputs"] == 1 and row["num_outputs"] == 2 and row["total_input_amount"] >= peel_value_q995 and row["_out_asym"] >= 2:
            return "peel_chain"
        if row["num_outputs"] >= fan_out_thresh:
            return "rapid_fan_out_layering"
        if row["num_inputs"] >= fan_in_thresh:
            return "fan_in_smurfing"
        if row["is_offhours"] and row["total_output_amount"] >= 1 and abs(row["total_output_amount"] - round(row["total_output_amount"])) < 0.02:
            return "round_amount_offhours"
        if row["src_ip_burst_count"] > ip_burst_thresh:
            return "shared_ip_burst"
        return "none"

    df["_motif"] = df.apply(classify, axis=1)
    thresholds = {
        "fan_out_thresh": round(fan_out_thresh, 2), "fan_in_thresh": round(fan_in_thresh, 2),
        "peel_value_q995": round(peel_value_q995, 4), "ip_burst_thresh": ip_burst_thresh,
    }
    return df, thresholds


TYPE_LABELS = {
    "peel_chain": "Peeling Chain", "round_amount_offhours": "Round-Amount Off-Hours",
    "rapid_fan_out_layering": "Rapid Fan-Out Layering", "fan_in_smurfing": "Fan-In Smurfing",
    "shared_ip_burst": "Shared-IP Burst", "none": "Unclassified Outlier",
}
TYPE_DESC = {
    "peel_chain": "Sequential 1-in/2-out cascade: a small amount is peeled off to a destination while the bulk of value is forwarded to a fresh change address — the classic laundering signature.",
    "round_amount_offhours": "Suspiciously round BTC value moved during low-activity hours, consistent with automated/scripted illicit transfers rather than organic spend behaviour.",
    "rapid_fan_out_layering": "A single input rapidly fans out to many outputs — a layering technique used to fragment funds across many wallets to defeat tracing.",
    "fan_in_smurfing": "Many small inputs converge into one transaction — classic smurfing/structuring pattern used to consolidate proceeds while staying under reporting thresholds.",
    "shared_ip_burst": "Multiple transactions broadcast from the same source IP in a short burst, touching many distinct wallets — indicative of a single operator running several addresses.",
    "none": "Statistically unusual transaction that did not match a known laundering motif but deviates significantly from population norms.",
}
