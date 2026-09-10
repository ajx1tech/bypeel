"""
Graph engine.

Builds the heterogeneous IP <-> Wallet <-> Transaction knowledge graph with
NetworkX, and resolves wallet addresses into logical entities via the
Common-Input-Ownership Heuristic (CIOH), implemented as a Union-Find
(Disjoint Set Union) structure for near-O(1) amortized merges — exactly the
"Graph Engine" module described in the problem statement.
"""
import networkx as nx
from collections import defaultdict


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def resolve_entities(df):
    """Union all input addresses spent together in the same transaction (CIOH)."""
    uf = UnionFind()
    for ins in df["input_list"]:
        for a in ins:
            uf.find(a)
        for i in range(1, len(ins)):
            uf.union(ins[0], ins[i])
    for outs in df["output_list"]:
        for a in outs:
            uf.find(a)

    entity_id_map = {}

    def eid(addr):
        root = uf.find(addr)
        if root not in entity_id_map:
            entity_id_map[root] = f"E{len(entity_id_map) + 1:04d}"
        return entity_id_map[root]

    entity_members = defaultdict(set)
    for addr in uf.parent:
        e = eid(addr)
        entity_members[e].add(addr)

    return uf, eid, entity_members


def build_networkx_graph(df, eid_fn):
    """
    Builds the full heterogeneous multigraph in NetworkX:
      node types: 'ip', 'wallet', 'tx'
      edge types: RELAYED (ip->tx), RELAYED_TO (tx->ip),
                  SPENT_FROM (wallet->tx), OUTPUT_TO (tx->wallet)
    Returns the graph plus a short() helper is left to caller.
    """
    G = nx.MultiDiGraph()
    for _, row in df.iterrows():
        tx_id = f"tx:{row['txid']}"
        G.add_node(tx_id, kind="tx", txid=row["txid"], timestamp=str(row["timestamp"]),
                   anomaly=int(row.get("is_anomaly", 0) or 0),
                   anomaly_type=row.get("_motif", "none"),
                   fee=float(row.get("fee", 0)),
                   total_in=float(row["total_input_amount"]),
                   total_out=float(row["total_output_amount"]))

        ip_s, ip_d = f"ip:{row['src_ip']}", f"ip:{row['dst_ip']}"
        G.add_node(ip_s, kind="ip", ip=row["src_ip"], country=row.get("src_country"))
        G.add_node(ip_d, kind="ip", ip=row["dst_ip"], country=row.get("dst_country"))
        G.add_edge(ip_s, tx_id, kind="RELAYED", anomaly=int(row.get("is_anomaly", 0) or 0))
        G.add_edge(tx_id, ip_d, kind="RELAYED_TO")

        for a, amt in zip(row["input_list"], row["input_amt_list"]):
            wid = f"w:{a}"
            G.add_node(wid, kind="wallet", addr=a, entity=eid_fn(a))
            G.add_edge(wid, tx_id, kind="SPENT_FROM", amount=float(amt),
                       anomaly=int(row.get("is_anomaly", 0) or 0))
        for a, amt in zip(row["output_list"], row["output_amt_list"]):
            wid = f"w:{a}"
            G.add_node(wid, kind="wallet", addr=a, entity=eid_fn(a))
            G.add_edge(tx_id, wid, kind="OUTPUT_TO", amount=float(amt),
                       anomaly=int(row.get("is_anomaly", 0) or 0))
    return G


def graph_to_frontend_json(G, df, eid_fn, max_fan=6, ambient_sample=45, seed=42):
    """
    Converts a (usually large) NetworkX graph into the compact node/edge JSON
    shape the bypeel.html frontend expects — capping high fan-in/fan-out
    transactions to `max_fan` explicit wallet nodes plus one aggregate "+N
    more" node, and sampling a small ambient set of non-flagged transactions
    for visual context (mirrors the client-side ingestion pipeline so the
    Python and JS ingestion paths produce visually consistent graphs).
    """
    import random
    rng = random.Random(seed)

    def short(a, n=6):
        return a if not isinstance(a, str) or len(a) <= n + 4 else f"{a[:n]}…{a[-4:]}"

    tx_rows = {f"tx:{r['txid']}": r for _, r in df.iterrows()}
    core_ids = [tid for tid, r in tx_rows.items() if int(r.get("is_anomaly", 0) or 0) == 1 or r.get("_motif", "none") != "none"]
    core_set = set(core_ids)

    # 1-hop forward chain extension: outputs of core tx reused as inputs elsewhere
    addr_in_idx = defaultdict(list)
    for tid, r in tx_rows.items():
        for a in r["input_list"]:
            addr_in_idx[a].append(tid)

    extra = set()
    for tid in core_ids:
        r = tx_rows[tid]
        for a in r["output_list"]:
            for j_tid in addr_in_idx.get(a, [])[:2]:
                if tx_rows[j_tid]["_ts"] > r["_ts"]:
                    extra.add(j_tid)

    normal_pool = [tid for tid in tx_rows if tid not in core_set and tid not in extra]
    ambient = set(rng.sample(normal_pool, min(ambient_sample, len(normal_pool)))) if normal_pool else set()

    keep = core_set | extra | ambient
    nodes, edges = {}, []

    def add_node(nid, **meta):
        if nid not in nodes:
            nodes[nid] = {"id": nid, **meta}
        return nodes[nid]

    for tid in keep:
        r = tx_rows[tid]
        anomaly = int(r.get("is_anomaly", 0) or 0)
        add_node(tid, type="tx", label=short(r["txid"], 8), txid=r["txid"], timestamp=str(r["timestamp"]),
                  anomaly=anomaly, anomaly_type=r.get("_motif", "none"), fee=float(r.get("fee", 0)),
                  script_type=r.get("script_type", "UNKNOWN"),
                  n_inputs=len(r["input_list"]), n_outputs=len(r["output_list"]),
                  total_in=round(float(r["total_input_amount"]), 6), total_out=round(float(r["total_output_amount"]), 6))
        ip_s, ip_d = f"ip:{r['src_ip']}", f"ip:{r['dst_ip']}"
        add_node(ip_s, type="ip", label=r["src_ip"], ip=r["src_ip"], country=r.get("src_country"), asn=str(r.get("src_asn")))
        add_node(ip_d, type="ip", label=r["dst_ip"], ip=r["dst_ip"], country=r.get("dst_country"), asn=str(r.get("dst_asn")))
        edges.append({"source": ip_s, "target": tid, "type": "RELAYED", "anomaly": anomaly})
        edges.append({"source": tid, "target": ip_d, "type": "RELAYED_TO"})

        ins = sorted(zip(r["input_list"], r["input_amt_list"]), key=lambda x: -x[1])
        outs = sorted(zip(r["output_list"], r["output_amt_list"]), key=lambda x: -x[1])

        for a, amt in ins[:max_fan]:
            wid = f"w:{a}"
            add_node(wid, type="wallet", label=short(a), addr=a, entity=eid_fn(a))
            edges.append({"source": wid, "target": tid, "type": "SPENT_FROM", "amount": round(amt, 6), "anomaly": anomaly})
        if len(ins) > max_fan:
            hidden = ins[max_fan:]
            agg_id = f"agg-in:{r['txid']}"
            add_node(agg_id, type="aggregate", label=f"+{len(hidden)} more inputs",
                      count=len(hidden), amount=round(sum(a for _, a in hidden), 6), direction="in")
            edges.append({"source": agg_id, "target": tid, "type": "SPENT_FROM",
                          "amount": round(sum(a for _, a in hidden), 6), "anomaly": anomaly, "aggregate": True})

        for a, amt in outs[:max_fan]:
            wid = f"w:{a}"
            add_node(wid, type="wallet", label=short(a), addr=a, entity=eid_fn(a))
            edges.append({"source": tid, "target": wid, "type": "OUTPUT_TO", "amount": round(amt, 6), "anomaly": anomaly})
        if len(outs) > max_fan:
            hidden = outs[max_fan:]
            agg_id = f"agg-out:{r['txid']}"
            add_node(agg_id, type="aggregate", label=f"+{len(hidden)} more outputs",
                      count=len(hidden), amount=round(sum(a for _, a in hidden), 6), direction="out")
            edges.append({"source": tid, "target": agg_id, "type": "OUTPUT_TO",
                          "amount": round(sum(a for _, a in hidden), 6), "anomaly": anomaly, "aggregate": True})

    return {"nodes": list(nodes.values()), "edges": edges}
