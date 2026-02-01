"""
Network Analysis for Public Procurement Fraud Detection.

This module implements graph-based methods for detecting:
1. Cartels (co-bidding networks)
2. Bid rigging rings (winner-loser patterns)
3. Monopolistic relationships (buyer-supplier networks)

Author: Roman Kachmar
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import combinations

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False

try:
    from community import community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False


@dataclass
class NetworkConfig:
    """Configuration for network analysis."""
    min_co_bids: int = 3  # Minimum co-bids for edge
    min_contracts: int = 3  # Minimum contracts for buyer-supplier edge
    min_community_size: int = 3  # Minimum community size to report


class NetworkAnalysisDetector:
    """
    Network-based anomaly detector for procurement data.

    Builds and analyzes multiple graphs:
    1. Co-bidding network: bidders who participate in same tenders
    2. Winner-loser network: directed graph of who beats whom
    3. Buyer-supplier network: contract relationships

    Usage:
        detector = NetworkAnalysisDetector(min_co_bids=3)
        results = detector.fit_detect(tenders, bids_df=bids)
        print(detector.summary())
    """

    def __init__(
        self,
        min_co_bids: int = 3,
        min_contracts: int = 3,
        min_community_size: int = 3,
        # Anomaly thresholds
        suspicious_min_degree: int = 10,
        suspicious_min_clustering: float = 0.7,
        rotation_min_ratio: float = 0.7,
        rotation_min_interactions: int = 5,
        monopoly_min_ratio: float = 0.9,
        monopoly_min_contracts: int = 20,
        # Optional features
        build_full_collusion: bool = False,
    ):
        """
        Initialize Network Analysis detector.

        Args:
            min_co_bids: Minimum co-bids for co-bidding edge
            min_contracts: Minimum contracts for buyer-supplier edge
            min_community_size: Minimum community size to consider
            suspicious_min_degree: Min degree for suspicious supplier
            suspicious_min_clustering: Min clustering for suspicious supplier
            rotation_min_ratio: Min rotation ratio for bid rotation flag
            rotation_min_interactions: Min total interactions for rotation
            monopoly_min_ratio: Min dominance ratio for monopoly flag
            monopoly_min_contracts: Min total contracts for monopoly flag
            build_full_collusion: Build full collusion graph (slow, optional)
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx package not installed. Run: pip install networkx")

        self.min_co_bids = min_co_bids
        self.min_contracts = min_contracts
        self.min_community_size = min_community_size

        # Anomaly thresholds
        self.suspicious_min_degree = suspicious_min_degree
        self.suspicious_min_clustering = suspicious_min_clustering
        self.rotation_min_ratio = rotation_min_ratio
        self.rotation_min_interactions = rotation_min_interactions
        self.monopoly_min_ratio = monopoly_min_ratio
        self.monopoly_min_contracts = monopoly_min_contracts

        # Optional features
        self.build_full_collusion = build_full_collusion

        # Graphs
        self.G_cobid = None  # Co-bidding graph
        self.G_winlose = None  # Winner-loser graph
        self.G_buyer_supplier = None  # Buyer-supplier graph
        self.G_full_collusion = None  # Full collusion graph (all combined)

        # Results
        self.cobid_communities = None
        self.cobid_metrics = None
        self.rotation_pairs = None
        self.monopolistic_pairs = None
        self.full_collusion_communities = None  # Communities in full graph
        self.full_collusion_metrics = None  # Node metrics in full graph
        self.results = None

    def fit_detect(
        self,
        tenders: pd.DataFrame,
        bids_df: Optional[pd.DataFrame] = None,
        bidders_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build networks and detect anomalies.

        Args:
            tenders: Tenders DataFrame
            bids_df: Bids DataFrame (required for co-bidding and winner-loser)
            bidders_df: Bidders DataFrame (optional, for names)

        Returns:
            DataFrame with network-based anomaly flags per tender
        """
        print(f"Processing {len(tenders):,} tenders...")

        # Filter to competitive tenders
        competitive = tenders[
            (tenders["procurement_method"].isin(["open", "selective"])) &
            (tenders["number_of_tenderers"] >= 2)
        ].copy()
        print(f"  Competitive tenders: {len(competitive):,}")

        if bids_df is not None:
            bids_competitive = bids_df[bids_df["tender_id"].isin(competitive["tender_id"])]
            print(f"  Bids in competitive: {len(bids_competitive):,}")

            # Build co-bidding network
            print("\nStep 1/5: Building co-bidding network...")
            self._build_cobid_network(bids_competitive)

            # Analyze co-bidding communities
            print("Step 2/5: Detecting communities...")
            self._analyze_cobid_communities()

            # Build winner-loser network
            print("Step 3/5: Building winner-loser network...")
            self._build_winlose_network(bids_competitive, competitive)

        # Build buyer-supplier network
        print("Step 4/5: Building buyer-supplier network...")
        self._build_buyer_supplier_network(tenders)

        # Build full collusion graph (combines all three) - OPTIONAL
        if self.build_full_collusion:
            print("Step 5/5: Building full collusion graph...")
            self._build_full_collusion_graph()
        else:
            print("Step 5/5: Skipping full collusion graph (disabled)")

        # Compute tender-level results
        print("\nComputing tender-level results...")
        self.results = self._compute_tender_results(tenders, bids_df)

        # Summary
        n_anomalies = self.results["network_anomaly"].sum() if "network_anomaly" in self.results.columns else 0
        print(f"\nNetwork Analysis complete!")
        print(f"  Tenders with network flags: {n_anomalies:,}")

        return self.results

    def _build_cobid_network(self, bids: pd.DataFrame) -> None:
        """Build co-bidding network from bids."""
        # Filter out NA bidder_ids
        bids_clean = bids[bids["bidder_id"].notna()].copy()

        co_bids = defaultdict(int)
        tender_bidders = bids_clean.groupby("tender_id")["bidder_id"].apply(list)

        for tender_id, bidder_list in tender_bidders.items():
            # Filter out any remaining NA values
            valid_bidders = [b for b in bidder_list if pd.notna(b)]
            unique_bidders = list(set(valid_bidders))
            if len(unique_bidders) >= 2:
                for pair in combinations(sorted(unique_bidders), 2):
                    co_bids[pair] += 1

        # Build graph
        self.G_cobid = nx.Graph()
        for (b1, b2), count in co_bids.items():
            if count >= self.min_co_bids:
                self.G_cobid.add_edge(b1, b2, weight=count)

        print(f"    Nodes: {self.G_cobid.number_of_nodes():,}")
        print(f"    Edges: {self.G_cobid.number_of_edges():,}")

    def _analyze_cobid_communities(self) -> None:
        """Detect communities and compute metrics in co-bidding network."""
        if self.G_cobid is None or self.G_cobid.number_of_nodes() == 0:
            self.cobid_communities = {}
            self.cobid_metrics = pd.DataFrame()
            return

        # Use igraph for faster computation if available
        if IGRAPH_AVAILABLE:
            self._analyze_with_igraph()
        else:
            self._analyze_with_networkx()

    def _analyze_with_igraph(self) -> None:
        """Fast community detection and metrics using igraph."""
        print("    Using igraph (fast)...")

        # Convert NetworkX to igraph
        # Get edges with weights
        edges = list(self.G_cobid.edges(data=True))
        nodes = list(self.G_cobid.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        g_ig = ig.Graph()
        g_ig.add_vertices(len(nodes))
        g_ig.vs["name"] = nodes

        edge_list = [(node_to_idx[e[0]], node_to_idx[e[1]]) for e in edges]
        weights = [e[2].get("weight", 1) for e in edges]
        g_ig.add_edges(edge_list)
        g_ig.es["weight"] = weights

        # Community detection (Louvain/Multilevel - very fast)
        communities = g_ig.community_multilevel(weights="weight")
        partition = {nodes[i]: communities.membership[i] for i in range(len(nodes))}
        self.cobid_communities = partition

        # Compute metrics (all fast in igraph)
        degrees = g_ig.degree()
        clustering = g_ig.transitivity_local_undirected(mode="zero")

        # Betweenness - igraph is much faster
        if len(nodes) > 10000:
            # Sample for very large graphs
            betweenness = g_ig.betweenness(weights="weight", cutoff=3)
        else:
            betweenness = g_ig.betweenness(weights="weight")

        # Normalize betweenness
        n = len(nodes)
        if n > 2:
            max_betweenness = (n - 1) * (n - 2) / 2
            betweenness = [b / max_betweenness if max_betweenness > 0 else 0 for b in betweenness]

        self.cobid_metrics = pd.DataFrame({
            "bidder_id": nodes,
            "community": [partition[n] for n in nodes],
            "degree": degrees,
            "clustering": clustering,
            "betweenness": betweenness,
        })

        n_communities = len(set(partition.values()))
        print(f"    Communities: {n_communities}")

    def _analyze_with_networkx(self) -> None:
        """Fallback community detection using NetworkX."""
        print("    Using NetworkX (slower)...")

        # Community detection
        if LOUVAIN_AVAILABLE:
            partition = community_louvain.best_partition(self.G_cobid, weight="weight")
        else:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(self.G_cobid, weight="weight"))
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i

        self.cobid_communities = partition

        # Compute metrics
        degrees = dict(self.G_cobid.degree())
        clustering = nx.clustering(self.G_cobid)

        # Betweenness (approximate for large graphs)
        if self.G_cobid.number_of_nodes() > 5000:
            betweenness = nx.betweenness_centrality(
                self.G_cobid, k=min(500, self.G_cobid.number_of_nodes())
            )
        else:
            betweenness = nx.betweenness_centrality(self.G_cobid)

        self.cobid_metrics = pd.DataFrame([
            {
                "bidder_id": node,
                "community": partition.get(node, -1),
                "degree": degrees.get(node, 0),
                "clustering": clustering.get(node, 0),
                "betweenness": betweenness.get(node, 0),
            }
            for node in self.G_cobid.nodes()
        ])

        n_communities = len(set(partition.values()))
        print(f"    Communities: {n_communities}")

    def _build_winlose_network(self, bids: pd.DataFrame, tenders: pd.DataFrame) -> None:
        """Build winner-loser directed network."""
        win_lose_edges = defaultdict(int)

        # Get winner for each tender
        tender_winners = tenders[["tender_id", "supplier_id"]].copy()
        tender_winners = tender_winners.rename(columns={"supplier_id": "winner_id"})

        bids_with_winner = bids.merge(tender_winners, on="tender_id", how="left")

        # Create winner -> loser edges
        for tender_id, group in bids_with_winner.groupby("tender_id"):
            winner_id = group["winner_id"].iloc[0]
            if pd.isna(winner_id) or winner_id is None:
                continue
            losers = group[group["bidder_id"] != winner_id]["bidder_id"].unique()
            for loser_id in losers:
                if pd.isna(loser_id) or loser_id is None:
                    continue
                win_lose_edges[(winner_id, loser_id)] += 1

        # Build directed graph
        self.G_winlose = nx.DiGraph()
        for (winner, loser), count in win_lose_edges.items():
            if count >= self.min_co_bids and winner is not None and loser is not None:
                self.G_winlose.add_edge(winner, loser, weight=count)

        print(f"    Nodes: {self.G_winlose.number_of_nodes():,}")
        print(f"    Edges: {self.G_winlose.number_of_edges():,}")

        # Find bid rotation (reciprocal edges)
        rotation_pairs = []
        for u, v, data in self.G_winlose.edges(data=True):
            if self.G_winlose.has_edge(v, u):
                reverse_weight = self.G_winlose[v][u]["weight"]
                if u < v:  # Avoid duplicates
                    rotation_ratio = min(data["weight"], reverse_weight) / max(data["weight"], reverse_weight)
                    rotation_pairs.append({
                        "bidder_1": u,
                        "bidder_2": v,
                        "wins_1_over_2": data["weight"],
                        "wins_2_over_1": reverse_weight,
                        "total_interactions": data["weight"] + reverse_weight,
                        "rotation_ratio": rotation_ratio,
                    })

        self.rotation_pairs = pd.DataFrame(rotation_pairs)
        if len(self.rotation_pairs) > 0:
            self.rotation_pairs = self.rotation_pairs.sort_values("rotation_ratio", ascending=False)
            n_suspicious = len(self.rotation_pairs[self.rotation_pairs["rotation_ratio"] >= 0.5])
            print(f"    Bid rotation pairs: {n_suspicious}")

    def _build_buyer_supplier_network(self, tenders: pd.DataFrame) -> None:
        """Build buyer-supplier bipartite network."""
        pair_stats = tenders.groupby(["buyer_id", "supplier_id"]).agg({
            "tender_id": "count",
            "award_value": "sum"
        }).reset_index()
        pair_stats.columns = ["buyer_id", "supplier_id", "contract_count", "total_value"]

        # Filter to significant relationships
        significant = pair_stats[pair_stats["contract_count"] >= self.min_contracts]

        # Build graph
        self.G_buyer_supplier = nx.Graph()
        for _, row in significant.iterrows():
            buyer_node = f"B_{row['buyer_id']}"
            supplier_node = f"S_{row['supplier_id']}"
            self.G_buyer_supplier.add_node(buyer_node, node_type="buyer")
            self.G_buyer_supplier.add_node(supplier_node, node_type="supplier")
            self.G_buyer_supplier.add_edge(
                buyer_node, supplier_node,
                weight=row["contract_count"],
                value=row["total_value"]
            )

        print(f"    Nodes: {self.G_buyer_supplier.number_of_nodes():,}")
        print(f"    Edges: {self.G_buyer_supplier.number_of_edges():,}")

        # Find monopolistic relationships
        buyer_totals = tenders.groupby("buyer_id")["tender_id"].count().reset_index()
        buyer_totals.columns = ["buyer_id", "total_contracts"]

        dominant = pair_stats.loc[pair_stats.groupby("buyer_id")["contract_count"].idxmax()]
        dominant = dominant.merge(buyer_totals, on="buyer_id")
        dominant["dominance_ratio"] = dominant["contract_count"] / dominant["total_contracts"]

        # Store all dominant pairs (filter later with configurable thresholds)
        self.monopolistic_pairs = dominant[
            dominant["dominance_ratio"] >= 0.5  # Store more, filter later
        ].sort_values("total_value", ascending=False)

        # Report with default loose threshold for info
        n_monopolistic = len(dominant[
            (dominant["dominance_ratio"] >= 0.8) &
            (dominant["total_contracts"] >= 10)
        ])
        print(f"    Monopolistic pairs (>=80%, >=10 contracts): {n_monopolistic}")

    def _build_full_collusion_graph(self) -> None:
        """
        Build full collusion graph combining all three networks.

        Nodes:
        - B_{buyer_id} - buyers
        - S_{supplier_id} - suppliers/bidders

        Edges:
        - contract: buyer <-> supplier (from buyer-supplier network)
        - cobid: supplier <-> supplier (from co-bidding network)
        - winlose: supplier <-> supplier (from winner-loser network)

        This unified graph enables detection of complex collusion schemes
        that span across buyers, suppliers, and their relationships.
        """
        self.G_full_collusion = nx.Graph()

        # 1. Add buyer-supplier edges (contract relationships)
        if self.G_buyer_supplier is not None:
            for u, v, data in self.G_buyer_supplier.edges(data=True):
                self.G_full_collusion.add_edge(u, v, edge_type="contract", **data)

        # 2. Add co-bidding edges (bidders who participate together)
        if self.G_cobid is not None:
            for u, v, data in self.G_cobid.edges(data=True):
                # Add S_ prefix if not already present
                node_u = f"S_{u}" if not str(u).startswith("S_") else u
                node_v = f"S_{v}" if not str(v).startswith("S_") else v

                # Add or update edge
                if self.G_full_collusion.has_edge(node_u, node_v):
                    self.G_full_collusion[node_u][node_v]["cobid_weight"] = data.get("weight", 1)
                else:
                    self.G_full_collusion.add_edge(node_u, node_v, edge_type="cobid", **data)

        # 3. Add winner-loser edges (competition relationships)
        if self.G_winlose is not None:
            for u, v, data in self.G_winlose.edges(data=True):
                node_u = f"S_{u}" if not str(u).startswith("S_") else u
                node_v = f"S_{v}" if not str(v).startswith("S_") else v

                if self.G_full_collusion.has_edge(node_u, node_v):
                    self.G_full_collusion[node_u][node_v]["winlose_weight"] = data.get("weight", 1)
                else:
                    self.G_full_collusion.add_edge(node_u, node_v, edge_type="winlose", **data)

        print(f"    Full collusion graph:")
        print(f"      Nodes: {self.G_full_collusion.number_of_nodes():,}")
        print(f"      Edges: {self.G_full_collusion.number_of_edges():,}")

        # Analyze communities in full graph
        if self.G_full_collusion.number_of_nodes() > 0:
            self._analyze_full_collusion_communities()

    def _analyze_full_collusion_communities(self) -> None:
        """Detect communities in the full collusion graph using igraph for speed."""
        if not IGRAPH_AVAILABLE:
            print("    igraph not available, skipping full collusion analysis")
            return

        print("    Detecting communities in full collusion graph...")

        # Convert to igraph
        nodes = list(self.G_full_collusion.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        g_ig = ig.Graph()
        g_ig.add_vertices(len(nodes))
        g_ig.vs["name"] = nodes
        g_ig.vs["type"] = ["buyer" if str(n).startswith("B_") else "supplier" for n in nodes]

        edges = list(self.G_full_collusion.edges(data=True))
        edge_list = [(node_to_idx[e[0]], node_to_idx[e[1]]) for e in edges]
        weights = [e[2].get("weight", 1) for e in edges]
        g_ig.add_edges(edge_list)
        g_ig.es["weight"] = weights

        # Community detection (Louvain/Multilevel)
        communities = g_ig.community_multilevel(weights="weight")
        partition = {nodes[i]: communities.membership[i] for i in range(len(nodes))}
        self.full_collusion_communities = partition

        # Count community sizes
        community_sizes = defaultdict(int)
        for node, comm_id in partition.items():
            community_sizes[comm_id] += 1

        # Find mixed communities (containing both buyers and suppliers)
        mixed_communities = []
        for comm_id in set(partition.values()):
            comm_nodes = [n for n, c in partition.items() if c == comm_id]
            buyers = [n for n in comm_nodes if str(n).startswith("B_")]
            suppliers = [n for n in comm_nodes if str(n).startswith("S_")]

            if len(buyers) > 0 and len(suppliers) > 0 and len(comm_nodes) >= self.min_community_size:
                mixed_communities.append({
                    "community_id": comm_id,
                    "total_nodes": len(comm_nodes),
                    "buyers": len(buyers),
                    "suppliers": len(suppliers),
                    "buyer_ids": [b.replace("B_", "") for b in buyers],
                    "supplier_ids": [s.replace("S_", "") for s in suppliers],
                })

        # Store metrics
        self.full_collusion_metrics = pd.DataFrame(mixed_communities)

        n_communities = len(set(partition.values()))
        n_mixed = len(mixed_communities)
        print(f"      Total communities: {n_communities}")
        print(f"      Mixed communities (buyers + suppliers): {n_mixed}")

    def get_collusion_communities(self, min_size: int = 3) -> pd.DataFrame:
        """
        Get potential collusion communities from full collusion graph.

        Returns communities that contain both buyers and suppliers,
        indicating potential complex collusion schemes.

        Args:
            min_size: Minimum community size

        Returns:
            DataFrame with community details
        """
        if self.full_collusion_metrics is None or len(self.full_collusion_metrics) == 0:
            return pd.DataFrame()

        return self.full_collusion_metrics[
            self.full_collusion_metrics["total_nodes"] >= min_size
        ].sort_values("total_nodes", ascending=False)

    def _compute_tender_results(
        self,
        tenders: pd.DataFrame,
        bids_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Compute tender-level network flags."""
        result = tenders[["tender_id", "buyer_id", "supplier_id"]].copy()

        # Flag 1: Supplier in suspicious community (high clustering + degree)
        if self.cobid_metrics is not None and len(self.cobid_metrics) > 0:
            suspicious_bidders = self.cobid_metrics[
                (self.cobid_metrics["degree"] >= self.suspicious_min_degree) &
                (self.cobid_metrics["clustering"] >= self.suspicious_min_clustering)
            ]["bidder_id"].tolist()

            result["network_suspicious_supplier"] = result["supplier_id"].isin(suspicious_bidders).astype(int)
            print(f"    Suspicious suppliers: {len(suspicious_bidders)}")
        else:
            result["network_suspicious_supplier"] = 0

        # Flag 2: Buyer-supplier monopolistic relationship (use stricter thresholds)
        if self.monopolistic_pairs is not None and len(self.monopolistic_pairs) > 0:
            strict_monopolistic = self.monopolistic_pairs[
                (self.monopolistic_pairs["dominance_ratio"] >= self.monopoly_min_ratio) &
                (self.monopolistic_pairs["total_contracts"] >= self.monopoly_min_contracts)
            ]
            monopolistic_set = set(zip(
                strict_monopolistic["buyer_id"],
                strict_monopolistic["supplier_id"]
            ))
            result["network_monopolistic"] = result.apply(
                lambda x: 1 if (x["buyer_id"], x["supplier_id"]) in monopolistic_set else 0,
                axis=1
            )
            print(f"    Strict monopolistic pairs: {len(strict_monopolistic)}")
        else:
            result["network_monopolistic"] = 0

        # Flag 3: Supplier in bid rotation pair (stricter criteria)
        if self.rotation_pairs is not None and len(self.rotation_pairs) > 0:
            strict_rotation = self.rotation_pairs[
                (self.rotation_pairs["rotation_ratio"] >= self.rotation_min_ratio) &
                (self.rotation_pairs["total_interactions"] >= self.rotation_min_interactions)
            ]
            rotation_bidders = set(
                strict_rotation["bidder_1"].tolist() +
                strict_rotation["bidder_2"].tolist()
            )
            result["network_rotation"] = result["supplier_id"].isin(rotation_bidders).astype(int)
            print(f"    Strict rotation pairs: {len(strict_rotation)}")
        else:
            result["network_rotation"] = 0

        # Combined network anomaly flag
        result["network_anomaly"] = (
            (result["network_suspicious_supplier"] == 1) |
            (result["network_monopolistic"] == 1) |
            (result["network_rotation"] == 1)
        ).astype(int)

        # Network score (0-3)
        result["network_score"] = (
            result["network_suspicious_supplier"] +
            result["network_monopolistic"] +
            result["network_rotation"]
        ) / 3

        return result[["tender_id", "network_suspicious_supplier", "network_monopolistic",
                       "network_rotation", "network_anomaly", "network_score"]]

    def get_cartel_candidates(self, min_size: int = 3) -> List[set]:
        """
        Get potential cartel groups from co-bidding communities.

        Args:
            min_size: Minimum group size

        Returns:
            List of sets, each containing bidder_ids in a community
        """
        if self.cobid_communities is None:
            return []

        # Group by community
        from collections import defaultdict
        communities = defaultdict(set)
        for bidder_id, comm_id in self.cobid_communities.items():
            communities[comm_id].add(bidder_id)

        # Filter by size
        return [members for members in communities.values() if len(members) >= min_size]

    def get_rotation_pairs(self, min_ratio: float = 0.5) -> pd.DataFrame:
        """Get bid rotation pairs with ratio above threshold."""
        if self.rotation_pairs is None or len(self.rotation_pairs) == 0:
            return pd.DataFrame()

        return self.rotation_pairs[self.rotation_pairs["rotation_ratio"] >= min_ratio]

    def get_monopolistic_relationships(self) -> pd.DataFrame:
        """Get monopolistic buyer-supplier relationships."""
        if self.monopolistic_pairs is None:
            return pd.DataFrame()
        return self.monopolistic_pairs

    def summary(self) -> pd.DataFrame:
        """Get summary of network analysis."""
        summary_data = []

        if self.G_cobid is not None:
            summary_data.extend([
                {"metric": "cobid_nodes", "value": self.G_cobid.number_of_nodes()},
                {"metric": "cobid_edges", "value": self.G_cobid.number_of_edges()},
            ])

        if self.cobid_communities is not None:
            n_communities = len(set(self.cobid_communities.values()))
            summary_data.append({"metric": "communities", "value": n_communities})

        if self.G_winlose is not None:
            summary_data.extend([
                {"metric": "winlose_nodes", "value": self.G_winlose.number_of_nodes()},
                {"metric": "winlose_edges", "value": self.G_winlose.number_of_edges()},
            ])

        if self.rotation_pairs is not None:
            n_rotation = len(self.rotation_pairs[self.rotation_pairs["rotation_ratio"] >= 0.5])
            summary_data.append({"metric": "rotation_pairs", "value": n_rotation})

        if self.monopolistic_pairs is not None:
            summary_data.append({"metric": "monopolistic_pairs", "value": len(self.monopolistic_pairs)})

        if self.G_full_collusion is not None:
            summary_data.extend([
                {"metric": "full_collusion_nodes", "value": self.G_full_collusion.number_of_nodes()},
                {"metric": "full_collusion_edges", "value": self.G_full_collusion.number_of_edges()},
            ])

        if self.full_collusion_communities is not None:
            n_total = len(set(self.full_collusion_communities.values()))
            summary_data.append({"metric": "full_collusion_communities", "value": n_total})

        if self.full_collusion_metrics is not None and len(self.full_collusion_metrics) > 0:
            n_mixed = len(self.full_collusion_metrics)
            summary_data.append({"metric": "mixed_communities", "value": n_mixed})

        if self.results is not None:
            n_flagged = self.results["network_anomaly"].sum()
            summary_data.append({"metric": "flagged_tenders", "value": n_flagged})

        return pd.DataFrame(summary_data)
