"""
Multi-format ingestion: CSV, JSON, and XML transaction metadata.

Normalizes arbitrary column names (case-insensitive) into the canonical
schema the rest of the pipeline expects, splits pipe-delimited multi-value
fields (input_addresses, input_amounts, ...), and de-duplicates by txid
across multiple uploaded files.
"""
import io
import json
import xml.etree.ElementTree as ET
import pandas as pd

CANONICAL_ALIASES = {
    "timestamp": ["timestamp", "time", "date", "ts"],
    "txid": ["txid", "tx_id", "transaction_id", "hash"],
    "src_ip": ["src_ip", "source_ip", "srcip"],
    "dst_ip": ["dst_ip", "dest_ip", "destination_ip", "dstip"],
    "src_country": ["src_country", "source_country"],
    "dst_country": ["dst_country", "destination_country"],
    "src_asn": ["src_asn", "source_asn"],
    "dst_asn": ["dst_asn", "destination_asn"],
    "src_asn_org": ["src_asn_org", "asn_org", "source_asn_org"],
    "input_addresses": ["input_addresses", "inputs", "sender_addresses"],
    "output_addresses": ["output_addresses", "outputs", "receiver_addresses"],
    "input_amounts": ["input_amounts", "sender_amounts"],
    "output_amounts": ["output_amounts", "receiver_amounts"],
    "fee": ["fee", "tx_fee"],
    "script_type": ["script_type", "scripttype"],
    "is_anomaly": ["is_anomaly", "label", "anomaly"],
    "anomaly_type": ["anomaly_type", "type", "motif"],
}


def _find_col(columns_lower, aliases):
    for a in aliases:
        if a in columns_lower:
            return columns_lower[a]
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame()
    for canon, aliases in CANONICAL_ALIASES.items():
        col = _find_col(cols_lower, aliases)
        out[canon] = df[col] if col else None
    return out


def parse_csv(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw_bytes))


def parse_json(raw_bytes: bytes) -> pd.DataFrame:
    data = json.loads(raw_bytes.decode("utf-8"))
    if isinstance(data, dict):
        for key in ("data", "transactions", "records", "rows"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    return pd.json_normalize(data)


def parse_xml(raw_bytes: bytes) -> pd.DataFrame:
    root = ET.fromstring(raw_bytes)
    # find the most common repeated child tag (the "record" element)
    tag_counts = {}
    for child in root.iter():
        tag_counts[child.tag] = tag_counts.get(child.tag, 0) + 1
    record_tag = max(tag_counts, key=tag_counts.get) if tag_counts else None
    records = root.findall(f".//{record_tag}") if record_tag else []
    rows = []
    for rec in records:
        row = dict(rec.attrib)
        for field in rec:
            texts = [t.text for t in rec.findall(field.tag) if t.text]
            if len(rec.findall(field.tag)) > 1:
                row[field.tag] = "|".join(texts)
            else:
                row[field.tag] = field.text
        rows.append(row)
    return pd.DataFrame(rows)


def load_file(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    lower = filename.lower()
    if lower.endswith(".csv"):
        df = parse_csv(raw_bytes)
    elif lower.endswith(".json"):
        df = parse_json(raw_bytes)
    elif lower.endswith(".xml"):
        df = parse_xml(raw_bytes)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    return _normalize_columns(df)


def load_many(files: list) -> pd.DataFrame:
    """files: list of (filename, raw_bytes). Returns deduped, normalized DataFrame."""
    frames = [load_file(name, data) for name, data in files]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "txid" in combined.columns:
        combined = combined.drop_duplicates(subset="txid", keep="first").reset_index(drop=True)
    # fill sane defaults for optional fields
    for col, default in [("src_country", "Unknown"), ("dst_country", "Unknown"),
                          ("src_asn_org", "Unknown"), ("script_type", "UNKNOWN"),
                          ("fee", 0), ("src_ip", "0.0.0.0"), ("dst_ip", "0.0.0.0")]:
        if col not in combined.columns:
            combined[col] = default
        combined[col] = combined[col].fillna(default)
    if "is_anomaly" in combined.columns:
        combined["is_anomaly"] = pd.to_numeric(combined["is_anomaly"], errors="coerce")
    return combined
