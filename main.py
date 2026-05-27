from z3 import *
import json


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


# INFO: for now, loading nodes and edges as separate data structures
# should unify these into some graph structure - depends on what is most compatible
# with pyz3. Maybe adjacency matrix?
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


class Worker:
    def __init__(self, worker_id: str, device_name: str, provides: dict, location: str):
        self.worker_id = worker_id
        self.device_name = device_name
        self.provides = provides
        self.location = location


def load_workers(fpath: str) -> list[Worker]:
    with open(fpath, "r") as file:
        worker_jsons = json.load(file)

    return {
        worker_json["worker_id"]: Worker(**worker_json) for worker_json in worker_jsons
    }


def load_workflows(fpath: str) -> list[Task]:
    with open(fpath, "r") as file:
        tasks_json = json.load(file)

    return [Task(**task) for task in tasks_json]


def main():
    workers = load_workers("data/workers.json")
    tasks = load_workflows("data/workflows.json")
    for w in workers:
        print(w)

    for t in tasks:
        print(t)


if __name__ == "__main__":
    main()
