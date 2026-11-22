"""Aerospace Engineering Knowledge Graph & Diagnostic Ontology (Phase 12.6).

Models semantic diagnostic relationships across the engine health hierarchy:
Engine -> Subsystem -> Sensor -> Abnormality -> Failure Mechanism -> Action

Features:
1. Triple-store ontology (Subject - Predicate - Object) with evidence provenance
2. Contextual subgraph extraction by Engine or Subsystem
3. Traceable reasoning paths from raw sensor signals to MRO actions
4. Interactive network graph layout generation for Plotly visualization
5. Backed by SQLite database (knowledge_graph_triples)
"""

from typing import Dict, List, Optional, Any, Tuple
import pathlib
import json
import numpy as np

from src.database_migrations import get_db_connection


class EngineeringKnowledgeGraph:
    """Aerospace Prognostic Ontology & Causal Knowledge Explorer."""

    def __init__(self, db_path: Optional[pathlib.Path] = None):
        self.db_path = db_path

    def get_all_triples(self) -> List[Dict[str, Any]]:
        """Retrieve all ontology triples from SQLite."""
        triples = []
        try:
            conn = get_db_connection(self.db_path)
            try:
                cur = conn.cursor()
                cur.execute("SELECT * FROM knowledge_graph_triples ORDER BY id ASC")
                for r in cur.fetchall():
                    triples.append({
                        "id": r["id"],
                        "subject": r["subject"],
                        "predicate": r["predicate"],
                        "object": r["object"],
                        "confidence": r["confidence"],
                        "provenance": r["provenance"]
                    })
            finally:
                conn.close()
        except Exception:
            pass
        return triples

    def query_subgraph(self, target_node: str) -> List[Dict[str, Any]]:
        """Extract adjacent 1-hop and 2-hop triples connected to target_node."""
        all_triples = self.get_all_triples()
        matched = []
        visited_nodes = {target_node.lower()}

        # 1-hop
        for t in all_triples:
            s_low = t["subject"].lower()
            o_low = t["object"].lower()
            if target_node.lower() in s_low or target_node.lower() in o_low:
                matched.append(t)
                visited_nodes.add(s_low)
                visited_nodes.add(o_low)

        # 2-hop
        for t in all_triples:
            if t in matched:
                continue
            s_low = t["subject"].lower()
            o_low = t["object"].lower()
            if any(n in s_low or n in o_low for n in visited_nodes):
                matched.append(t)

        return matched

    def build_network_layout(self, triples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate 2D node coordinates and edge traces for Plotly graph rendering."""
        # Collect unique nodes
        nodes = set()
        for t in triples:
            nodes.add(t["subject"])
            nodes.add(t["object"])

        node_list = sorted(list(nodes))
        num_nodes = len(node_list)

        # Circular layout positioning
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False) if num_nodes > 0 else []
        node_pos = {}
        for idx, n in enumerate(node_list):
            node_pos[n] = (float(np.cos(angles[idx]) * 10.0), float(np.sin(angles[idx]) * 10.0))

        # Node categories for coloring
        node_colors = []
        for n in node_list:
            if "Engine" in n:
                node_colors.append("#00B8D9")  # Cyan
            elif any(sub in n for sub in ["Compressor", "Turbine", "Combustor", "Fan"]):
                node_colors.append("#21C77A")  # Green
            elif any(s in n for s in ["T30", "T48", "Ps30", "EGT", "T50"]):
                node_colors.append("#FFB547")  # Amber
            elif any(m in n for m in ["Wash", "Borescope", "Inspection", "Teardown"]):
                node_colors.append("#FF5252")  # Red
            else:
                node_colors.append("#9BA9B8")  # Grey

        # Edge lines
        edge_x = []
        edge_y = []
        edge_text = []

        for t in triples:
            x0, y0 = node_pos[t["subject"]]
            x1, y1 = node_pos[t["object"]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"{t['subject']} -> [{t['predicate']}] -> {t['object']} (Conf: {t['confidence']})")

        return {
            "node_list": node_list,
            "node_x": [node_pos[n][0] for n in node_list],
            "node_y": [node_pos[n][1] for n in node_list],
            "node_colors": node_colors,
            "edge_x": edge_x,
            "edge_y": edge_y,
            "edge_text": edge_text,
            "triples_count": len(triples)
        }
