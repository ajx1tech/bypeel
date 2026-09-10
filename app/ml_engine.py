"""
ML scoring + explainability engine.

Loads the persisted `models/isolation_forest.pkl` bundle (an IsolationForest
+ StandardScaler + feature_names, saved with joblib) and scores every
ingested transaction. Explanations are produced with SHAP's TreeExplainer
when the `shap` package is available (fast, exact for tree ensembles); if
`shap` is not installed the engine falls back to a permutation-style
approximate attribution so the pipeline still runs (fully offline, no
network calls either way).
"""
import os
import numpy as np
import joblib

from .features import FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "isolation_forest.pkl")

FEATURE_LABELS = {
    "num_inputs": "Input Count (Fan-In)", "num_outputs": "Output Count (Fan-Out)",
    "total_input_amount": "Total Input Amount", "total_output_amount": "Total Output Amount",
    "fee_ratio": "Fee Ratio", "output_input_ratio": "Output/Input Ratio",
    "hour_of_day": "Hour of Day", "is_offhours": "Off-Hours Timing",
    "src_ip_freq": "Source IP Frequency", "src_asn_freq": "Source ASN Frequency",
    "src_country_freq": "Source Country Frequency", "src_is_hosting_asn": "Source Hosting ASN",
    "dst_is_hosting_asn": "Destination Hosting ASN", "src_country_unknown": "Unknown Source Country",
    "ip_wallet_diversity": "IP↔Wallet Diversity", "primary_input_wallet_freq": "Primary Wallet Frequency",
    "src_ip_burst_count": "IP Burst Count",
}


class MLEngine:
    def __init__(self, model_path=MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_names = bundle.get("feature_names", FEATURE_NAMES)
        self._explainer = None
        try:
            import shap
            self._explainer = shap.TreeExplainer(self.model)
            self._shap = shap
        except Exception:
            self._explainer = None

    def score(self, X_df):
        """Returns (raw_scores, normalized_0_1_scores) — higher = more anomalous."""
        X = X_df[self.feature_names]
        Xs = self.scaler.transform(X)
        # IsolationForest.score_samples: higher = more normal. Flip + normalize to 0..1.
        raw = -self.model.score_samples(Xs)
        lo, hi = raw.min(), raw.max()
        norm = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
        return raw, norm, Xs

    def explain(self, Xs, top_n=4):
        """
        Returns a list (len = n_rows) of lists of {feature, raw, value, z, direction,
        contribution_pct} dicts — TreeSHAP-based when available, else a
        median/MAD z-score fallback so the UI's explainability cards always work.
        """
        n = Xs.shape[0]
        if self._explainer is not None:
            try:
                shap_values = self._explainer.shap_values(Xs)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                out = []
                for i in range(n):
                    row_vals = shap_values[i]
                    order = np.argsort(-np.abs(row_vals))[:top_n]
                    total = sum(abs(row_vals[j]) for j in order) or 1.0
                    feats = []
                    for j in order:
                        name = self.feature_names[j]
                        feats.append({
                            "feature": FEATURE_LABELS.get(name, name), "raw": name,
                            "value": round(float(Xs[i, j]), 4), "z": round(float(row_vals[j]) * 10, 2),
                            "direction": "above" if row_vals[j] < 0 else "below",  # more negative shap => more anomalous contribution
                            "median": 0.0,
                            "contribution_pct": round(abs(row_vals[j]) / total * 100, 1),
                        })
                    out.append(feats)
                return out
            except Exception:
                pass  # fall through to z-score fallback

        # Fallback: median/MAD z-score per feature across the batch
        med = np.median(Xs, axis=0)
        mad = np.median(np.abs(Xs - med), axis=0)
        mad = np.where(mad > 0, mad, 1e-6)
        z = (Xs - med) / (1.4826 * mad)
        out = []
        for i in range(n):
            order = np.argsort(-np.abs(z[i]))[:top_n]
            total = sum(abs(z[i, j]) for j in order) or 1.0
            feats = []
            for j in order:
                name = self.feature_names[j]
                feats.append({
                    "feature": FEATURE_LABELS.get(name, name), "raw": name,
                    "value": round(float(Xs[i, j]), 4), "z": round(float(z[i, j]), 2),
                    "direction": "above" if z[i, j] >= 0 else "below", "median": 0.0,
                    "contribution_pct": round(abs(z[i, j]) / total * 100, 1),
                })
            out.append(feats)
        return out
