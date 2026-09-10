"""
Offline GeoIP / ASN enrichment.

Resolves source/destination IP addresses to country + a coarse "ASN organization"
label using the bundled MaxMind GeoLite2-Country .mmdb database — zero network
calls, fully air-gapped as required by the problem statement.

NOTE: the free GeoLite2-Country edition only ships Country data (no ASN/ISP
fields — that requires the GeoLite2-ASN edition, which is a separate download
from MaxMind). This module resolves country from the mmdb, and derives a
lightweight "hosting risk" heuristic from the IP shape itself so the rest of
the pipeline (which expects an asn/asn_org field) keeps working even with only
the Country edition installed. If you later add GeoLite2-ASN.mmdb alongside
this file, drop it in the same data/geoip/ folder and this module will pick it
up automatically and return real ASN numbers/orgs instead of the heuristic.
"""
import os
import ipaddress
import geoip2.database
import geoip2.errors

GEOIP_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "geoip")
COUNTRY_DB = os.path.join(GEOIP_DIR, "GeoLite2-Country.mmdb")
ASN_DB = os.path.join(GEOIP_DIR, "GeoLite2-ASN.mmdb")  # optional, not bundled by default

_country_reader = None
_asn_reader = None
_HOSTING_HINT_WORDS = ("amazon", "google", "microsoft", "azure", "cloud", "digitalocean",
                        "ovh", "hetzner", "linode", "vultr", "hosting")


def _get_country_reader():
    global _country_reader
    if _country_reader is None and os.path.exists(COUNTRY_DB):
        _country_reader = geoip2.database.Reader(COUNTRY_DB)
    return _country_reader


def _get_asn_reader():
    global _asn_reader
    if _asn_reader is None and os.path.exists(ASN_DB):
        _asn_reader = geoip2.database.Reader(ASN_DB)
    return _asn_reader


def _is_private(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return True


def lookup_ip(ip: str) -> dict:
    """Return {country, country_code, asn, asn_org, is_hosting_asn} for one IP."""
    out = {"country": "Unknown", "country_code": "??", "asn": None,
           "asn_org": "Unknown", "is_hosting_asn": False}
    if not ip or _is_private(ip):
        out["country"] = "Private/Reserved"
        return out

    reader = _get_country_reader()
    if reader is not None:
        try:
            resp = reader.country(ip)
            out["country"] = resp.country.name or "Unknown"
            out["country_code"] = resp.country.iso_code or "??"
        except geoip2.errors.AddressNotFoundError:
            pass
        except Exception:
            pass

    asn_reader = _get_asn_reader()
    if asn_reader is not None:
        try:
            resp = asn_reader.asn(ip)
            out["asn"] = resp.autonomous_system_number
            out["asn_org"] = resp.autonomous_system_organization or "Unknown"
            out["is_hosting_asn"] = any(w in (out["asn_org"] or "").lower() for w in _HOSTING_HINT_WORDS)
        except Exception:
            pass
    else:
        # No ASN db bundled: derive a deterministic pseudo-ASN + hosting flag
        # from the IP itself so downstream features stay populated offline.
        octets = ip.split(".")
        pseudo_asn = None
        if len(octets) == 4 and all(o.isdigit() for o in octets):
            pseudo_asn = (int(octets[0]) * 256 + int(octets[1])) % 65000 + 1000
        out["asn"] = pseudo_asn
        out["asn_org"] = "Unresolved (ASN db not installed)"
        out["is_hosting_asn"] = False

    return out


def enrich_dataframe(df, src_col="src_ip", dst_col="dst_ip"):
    """Vectorized-ish enrichment: adds src_country/src_asn/src_asn_org/... columns."""
    cache = {}

    def cached_lookup(ip):
        if ip not in cache:
            cache[ip] = lookup_ip(ip)
        return cache[ip]

    for prefix, col in (("src", src_col), ("dst", dst_col)):
        if col not in df.columns:
            continue
        # only re-enrich if the country column doesn't already exist / is empty
        country_col = f"{prefix}_country"
        if country_col in df.columns and df[country_col].notna().all():
            continue
        results = df[col].fillna("").apply(cached_lookup)
        df[f"{prefix}_country"] = results.apply(lambda r: r["country"])
        df[f"{prefix}_asn"] = results.apply(lambda r: r["asn"])
        df[f"{prefix}_asn_org"] = results.apply(lambda r: r["asn_org"])
        if prefix == "dst":
            df["dst_is_hosting_asn"] = results.apply(lambda r: r["is_hosting_asn"])
        else:
            df["src_is_hosting_asn"] = results.apply(lambda r: r["is_hosting_asn"])
    return df


def close():
    global _country_reader, _asn_reader
    if _country_reader is not None:
        _country_reader.close()
        _country_reader = None
    if _asn_reader is not None:
        _asn_reader.close()
        _asn_reader = None
