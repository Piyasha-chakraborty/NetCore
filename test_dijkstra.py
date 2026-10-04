"""Unit tests: our Dijkstra must agree with NetworkX. Run:  python test_dijkstra.py"""
import random, unittest
import networkx as nx
from netcore import dijkstra, Sim, candidate_paths, SCENARIOS


class TestDijkstra(unittest.TestCase):
    def test_random_graphs(self):
        random.seed(1)
        for _ in range(300):
            n = random.randint(4, 14)
            G = nx.gnp_random_graph(n, 0.35, seed=random.randint(0, 10**6), directed=True)
            for u, v in G.edges: G[u][v]["w"] = random.random() * 10
            adj = {u: list(G.successors(u)) for u in G.nodes}
            s, d = random.sample(list(G.nodes), 2)
            path, cost = dijkstra(adj, s, d, lambda u, v: G[u][v]["w"])
            if nx.has_path(G, s, d):
                ref = nx.dijkstra_path_length(G, s, d, weight="w")
                self.assertAlmostEqual(cost, ref, places=9)
                self.assertEqual(path[0], s); self.assertEqual(path[-1], d)
            else:
                self.assertIsNone(path)

    def test_known_graph(self):
        adj = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
        w = {("A", "B"): 1, ("B", "D"): 5, ("A", "C"): 2, ("C", "D"): 1}
        path, cost = dijkstra(adj, "A", "D", lambda u, v: w[(u, v)])
        self.assertEqual(path, ["A", "C", "D"]); self.assertEqual(cost, 3)

    def test_in_simulator_cost_matches_networkx(self):
        for sc in SCENARIOS:
            s = Sim(sc, 7); s.run()
            for t in (50, 150, 250):
                df, ours, theirs = candidate_paths(s, "S1", "S6", t, 200.0)
                cost = lambda p: df[df.path == " > ".join(p)].total.iloc[0]
                self.assertAlmostEqual(cost(ours), cost(theirs), places=9)
                self.assertEqual(ours, df.sort_values("total").iloc[0].path.split(" > "))


if __name__ == "__main__":
    unittest.main(verbosity=2)
