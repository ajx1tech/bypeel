"""
Feature engineering for the anomaly-detection model.

Produces exactly the 17 columns the bundled `models/isolation_forest.pkl`
was trained on (see the dict's `feature_names` key):

    num_inputs, num_outputs, total_input_amount, total_output_amount,
    fee_ratio, output_input_ratio, hour_of_day, is_offhours,
    src_ip_freq, src_asn_freq, src_country_freq,
    src_is_hosting_asn, dst_is_hosting_asn, src_country_unknown,
    ip_wallet_diversity, primary_input_wallet_freq, src_ip_burst_count

All features are computed from the ingested corpus itself (population
statistics), so the model generalizes to whatever dataset is uploaded rather
than being hard-coded to any one synthetic generator.
"""
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "num_inputs", "num_outputs", "total_input_amount", "total_output_amount",
    "fee_ratio", "output_input_ratio", "hour_of_day", "is_offhours",
    "src_ip_freq", "src_asn_freq", "src_country_freq",
    "src_is_hosting_asn", "dst_is_hosting_asn", "src_country_unknown",
    "ip_wallet_diversity", "primary_input_wallet_freq", "src_ip_burst_count",
]

IP_BURST_WINDOW_MINUTES = 15


def _split_list(s):
    if isinstance(s, list):
        return s
    if pd.isna(s) or s == "":
        return []
    return str(s).split("|")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mutates/returns df with all engineered columns + raw parsed list columns."""
    df = df.copy()
    df["input_list"] = df["input_addresses"].apply(_split_list)
    df["output_list"] = df["output_addresses"].apply(_split_list)
    df["input_amt_list"] = df["input_amounts"].apply(
        lambda s: [float(x) for x in _split_list(s)] if _split_list(s) else [])
    df["output_amt_list"] = df["output_amounts"].apply(
        lambda s: [float(x) for x in _split_list(s)] if _split_list(s) else [])

    df["num_inputs"] = df["input_list"].apply(len)
    df["num_outputs"] = df["output_list"].apply(len)
    df["total_input_amount"] = df["input_amt_list"].apply(sum)
    df["total_output_amount"] = df["output_amt_list"].apply(sum)
    df["fee"] = pd.to_numeric(df.get("fee", 0), errors="coerce").fillna(0.0)

    df["fee_ratio"] = df["fee"] / df["total_input_amount"].replace(0, np.nan)
    df["fee_ratio"] = df["fee_ratio"].fillna(0.0).clip(0, 5)

    df["output_input_ratio"] = df["total_output_amount"] / df["total_input_amount"].replace(0, np.nan)
    df["output_input_ratio"] = df["output_input_ratio"].fillna(1.0).clip(0, 5)

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["_ts"] = ts
    df["hour_of_day"] = ts.dt.hour.fillna(12).astype(int)
    df["is_offhours"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)

    # population frequency features
    df["src_ip_freq"] = df.groupby("src_ip")["src_ip"].transform("count")
    if "src_asn" in df.columns:
        df["src_asn_freq"] = df.groupby("src_asn")["src_asn"].transform("count")
    else:
        df["src_asn_freq"] = 0
    if "src_country" in df.columns:
        df["src_country_freq"] = df.groupby("src_country")["src_country"].transform("count")
        df["src_country_unknown"] = df["src_country"].isin(["Unknown", "Private/Reserved", None]).astype(int)
    else:
        df["src_country_freq"] = 0
        df["src_country_unknown"] = 1

    df["src_is_hosting_asn"] = df["src_is_hosting_asn"] if "src_is_hosting_asn" in df.columns else False
    df["src_is_hosting_asn"] = df["src_is_hosting_asn"].fillna(False).astype(int)
    df["dst_is_hosting_asn"] = df["dst_is_hosting_asn"] if "dst_is_hosting_asn" in df.columns else False
    df["dst_is_hosting_asn"] = df["dst_is_hosting_asn"].fillna(False).astype(int)

    # IP <-> wallet diversity: how many distinct wallets has this src_ip touched overall
    ip_wallets = {}
    for ip, ins, outs in zip(df["src_ip"], df["input_list"], df["output_list"]):
        s = ip_wallets.setdefault(ip, set())
        s.update(ins)
        s.update(outs)
    df["ip_wallet_diversity"] = df["src_ip"].map(lambda ip: len(ip_wallets.get(ip, [])))

    # primary input wallet frequency: how often does this tx's first input address
    # appear as an input anywhere else in the corpus
    wallet_input_freq = {}
    for ins in df["input_list"]:
        for a in ins:
            wallet_input_freq[a] = wallet_input_freq.get(a, 0) + 1
    df["primary_input_wallet_freq"] = df["input_list"].apply(
        lambda ins: wallet_input_freq.get(ins[0], 0) if ins else 0)

    # burst count: # of tx from the same src_ip within +/- IP_BURST_WINDOW_MINUTES
    df = df.sort_values("_ts").reset_index(drop=True)
    burst_counts = np.zeros(len(df), dtype=int)
    window = pd.Timedelta(minutes=IP_BURST_WINDOW_MINUTES)
    for ip, group in df.groupby("src_ip"):
        times = group["_ts"].values
        idx = group.index.to_numpy()
        times_s = pd.Series(times)
        for i, t in enumerate(times):
            lo = t - window.to_timedelta64()
            hi = t + window.to_timedelta64()
            cnt = int(((times_s >= lo) & (times_s <= hi)).sum())
            burst_counts[idx[i]] = cnt
    df["src_ip_burst_count"] = burst_counts

    return df


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract just the ordered model-input columns."""
    return df[FEATURE_NAMES].astype(float)
