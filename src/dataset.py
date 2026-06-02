import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Node:
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        processing_time: int,
        requires: dict,
        compatible_workers: list[str],
    ):
        self.id = id
        self.name = name
        self.desc = description
        self.proc_time = processing_time
        self.requires = requires
        self.compatible_workers = compatible_workers


class Edge:
    def __init__(self, src: Node, dst: Node, data: str):
        self.src = src
        self.dst = dst
        self.data = data


class Task:
    def __init__(
        self,
        task_id: str,
        task_description: str,
        dag_nodes: list[dict],
        dag_edges: list[dict],
        workers: list[str],
    ):
        self.task_id = task_id
        self.task_desc = task_description
        self.nodes = self._get_nodes(dag_nodes)
        self.edges = self._get_edges(dag_edges, self.nodes)
        self.workers = workers

    def _get_nodes(self, dag_node_json: list[dict]) -> dict[str, Node]:
        # nodes should be lookup hash table for fast retrieval when creating edge list
        return {node["id"]: Node(**node) for node in dag_node_json}

    # is a list here really a useful way to encode edges?
    def _get_edges(
        self, dag_edge_json: list[dict], nodes: dict[str, Node]
    ) -> list[Edge]:
        edges = []
        for edge in dag_edge_json:
            src_node = nodes.get(edge.get("from"))
            dst_node = nodes.get(edge.get("to"))
            edges.append(Edge(src_node, dst_node, edge["data"]))
        return edges

    # INFO: llm-generated prototype viz code
    def render_dag(self, output_path: str | None = None) -> str:
        """Render the task's DAG as an image and save it to disk.

        Parameters
        ----------
        output_path : str | None
            File path for the output image (e.g. ``"dag.png"``).
            If None, defaults to ``"<task_id>_dag.png"`` in the current directory.
            The format is inferred from the file extension (png, pdf, svg, etc.).

        Returns
        -------
        str
            The absolute path of the saved figure.
        """

        if output_path is None:
            output_path = f"{self.task_id}_dag.png"

        # ------------------------------------------------------------------
        # Build the networkx DiGraph
        # ------------------------------------------------------------------
        G = nx.DiGraph()

        for nid, node in self.nodes.items():
            G.add_node(
                nid, label=node.name, proc_time=node.proc_time, requires=node.requires
            )

        for edge in self.edges:
            if edge.src is None or edge.dst is None:
                continue
            data_label = (
                ", ".join(edge.data) if isinstance(edge.data, list) else str(edge.data)
            )
            G.add_edge(edge.src.id, edge.dst.id, data=data_label)

        # ------------------------------------------------------------------
        # Assign layers via longest-path distance from sources (topological)
        # ------------------------------------------------------------------
        topo_order = list(nx.topological_sort(G))
        layer: dict[str, int] = {}
        for nid in topo_order:
            preds = list(G.predecessors(nid))
            if not preds:
                layer[nid] = 0
            else:
                layer[nid] = max(layer[p] + 1 for p in preds)

        nx.set_node_attributes(G, layer, "subset")

        # ------------------------------------------------------------------
        # Colour nodes by resource tier
        # ------------------------------------------------------------------
        palette = {
            "cpu_light": "#6EC6FF",  # CPU ≤ 2
            "cpu_heavy": "#1565C0",  # CPU > 2, no GPU
            "gpu": "#AB47BC",  # GPU > 0
        }

        node_colours = []
        for nid in G.nodes:
            req = self.nodes[nid].requires
            gpu = req.get("GPU", 0)
            cpu = req.get("CPU", 0)
            if gpu > 0:
                node_colours.append(palette["gpu"])
            elif cpu > 2:
                node_colours.append(palette["cpu_heavy"])
            else:
                node_colours.append(palette["cpu_light"])

        # ------------------------------------------------------------------
        # Size nodes proportional to processing time
        # ------------------------------------------------------------------
        proc_times = [self.nodes[nid].proc_time for nid in G.nodes]
        max_pt = max(proc_times) if proc_times else 1
        node_sizes = [800 + 2200 * (pt / max_pt) for pt in proc_times]

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------
        pos = nx.multipartite_layout(G, subset_key="subset", align="horizontal")
        # Flip y so that sources are at the top
        max_y = max(v[1] for v in pos.values()) if pos else 0
        pos = {k: (v[0], max_y - v[1]) for k, v in pos.items()}

        # ------------------------------------------------------------------
        # Draw
        # ------------------------------------------------------------------
        num_layers = max(layer.values()) + 1 if layer else 1
        fig_height = max(6, 1.8 * num_layers)
        num_nodes_widest = max(
            sum(1 for l in layer.values() if l == lv) for lv in range(num_layers)
        )
        fig_width = max(10, 2.5 * num_nodes_widest)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")

        # Edges
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color="#90A4AE",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=16,
            width=1.5,
            connectionstyle="arc3,rad=0.08",
            min_source_margin=18,
            min_target_margin=18,
        )

        # Nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colours,
            node_size=node_sizes,
            edgecolors="#37474F",
            linewidths=1.5,
        )

        # Labels — show name + processing time
        labels = {}
        for nid in G.nodes:
            node = self.nodes[nid]
            req_parts = [f"{k}:{v}" for k, v in node.requires.items() if v > 0]
            req_str = " | ".join(req_parts)
            labels[nid] = f"{node.name}\n({node.proc_time}s, {req_str})"

        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            ax=ax,
            font_size=7,
            font_weight="bold",
            font_color="#212121",
        )

        # Edge labels (data transfer)
        edge_labels = nx.get_edge_attributes(G, "data")
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=5.5,
            font_color="#546E7A",
            bbox=dict(boxstyle="round,pad=0.15", fc="#ECEFF1", ec="none", alpha=0.85),
            rotate=False,
        )

        # Legend
        legend_handles = [
            mpatches.Patch(color=palette["cpu_light"], label="CPU ≤ 2"),
            mpatches.Patch(color=palette["cpu_heavy"], label="CPU > 2 (no GPU)"),
            mpatches.Patch(color=palette["gpu"], label="GPU required"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            fontsize=8,
            framealpha=0.9,
            edgecolor="#B0BEC5",
        )

        ax.set_title(
            f"DAG: {self.task_id}",
            fontsize=13,
            fontweight="bold",
            color="#263238",
            pad=14,
        )
        ax.axis("off")
        fig.tight_layout()

        fig.savefig(
            output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        plt.close(fig)

        abs_path = os.path.abspath(output_path)
        print(f"[dag] Saved DAG figure to {abs_path}")
        return abs_path


class Worker:
    def __init__(self, worker_id: str, device_name: str, provides: dict, location: str):
        self.worker_id = worker_id
        self.device_name = device_name
        self.provides = provides
        self.location = location
