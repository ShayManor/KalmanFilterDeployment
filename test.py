import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch


import sys

sys.path.append(
    r"C:\Users\zrz11\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages"
)


def create_fixed_graph():
    G = nx.DiGraph()

    G.add_node("V(0)", pos=(0, 0))
    G.add_node("Z(0)", pos=(1, 0))
    G.add_node("X(0)", pos=(1, 1))

    G.add_node("V(1)", pos=(2, 0))
    G.add_node("Z(1)", pos=(3, 0))
    G.add_node("X(1)", pos=(3, 1))

    G.add_node("V(2)", pos=(4, 0))
    G.add_node("Z(2)", pos=(5, 0))
    G.add_node("X(2)", pos=(5, 1))

    G.add_node("W(0)", pos=(3, 2))
    G.add_node("W(1)", pos=(5, 2))

    G.add_edge("V(0)", "Z(0)")
    G.add_edge("X(0)", "Z(0)")
    G.add_edge("V(1)", "Z(1)")
    G.add_edge("X(1)", "Z(1)")
    G.add_edge("V(2)", "Z(2)")
    G.add_edge("X(2)", "Z(2)")
    G.add_edge("X(0)", "X(1)")
    G.add_edge("X(1)", "X(2)")
    G.add_edge("W(0)", "X(1)")
    G.add_edge("W(1)", "X(2)")

    return G


def adjust_edge_coords(pos, source, target, node_radius):
    sx, sy = pos[source]
    tx, ty = pos[target]
    dx, dy = tx - sx, ty - sy
    dist = (dx**2 + dy**2) ** 0.5
    if dist == 0:
        return (sx, sy), (tx, ty)
    nx_dir = dx / dist
    ny_dir = dy / dist
    start = (sx + nx_dir * node_radius, sy + ny_dir * node_radius)
    end = (tx - nx_dir * node_radius, ty - ny_dir * node_radius)
    return start, end


def draw_fixed_graph(G):
    pos = nx.get_node_attributes(G, "pos")
    fig, ax = plt.subplots(figsize=(16, 6.5))

    orange_nodes = ["V(0)", "V(1)", "V(2)", "W(0)", "W(1)", "X(0)"]
    blue_nodes = ["Z(0)", "Z(1)", "Z(2)", "X(1)", "X(2)"]

    orange = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=orange_nodes,
        node_color="#ed841b",
        node_size=3200,
        edgecolors="black",
    )
    orange.set_zorder(1)

    blue_outer = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=blue_nodes,
        node_color="#00afee",
        node_size=3200,
        edgecolors="black",
        linewidths=1,
    )
    blue_outer.set_zorder(1)

    blue_inner = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=blue_nodes,
        node_color="#00afee",
        node_size=2700,
        edgecolors="black",
        linewidths=1,
    )
    blue_inner.set_zorder(2)

    labels = nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    for label in labels.values():
        label.set_zorder(3)

    node_radius = 0.155

    for source, target in G.edges():
        start, end = adjust_edge_coords(pos, source, target, node_radius)
        arrow = FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=15, color="black", zorder=4
        )
        ax.add_patch(arrow)

    ax.set_axis_off()
    plt.show()


G = create_fixed_graph()
draw_fixed_graph(G)
