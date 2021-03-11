import networkx as nx
import numpy as np


def main(filename="sim_matrix.obj"):
    dt = [('len', float)]
    A = np.load(filename, allow_pickle=True)[3:, 3:] / 50
    A = A.view(dt)
    G = nx.from_numpy_matrix(A)
    # G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), ["f1", "f2", "f3", "n1", "n2", "n3", "n4", "n5"])))
    # G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), ["f1", "f2", "f3"])))
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), ["n1", "n2", "n3", "n4", "n5"])))

    G = nx.drawing.nx_agraph.to_agraph(G)

    G.node_attr.update(color="red", style="filled")
    G.edge_attr.update(color="blue", width="2.0")

    G.draw('/tmp/out.png', format='png', prog='neato')


if __name__ == "__main__":
    main()
