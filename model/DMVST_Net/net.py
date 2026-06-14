import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from tslearn.metrics import dtw
from node2vec import Node2Vec

def visualize_semantic_graph(G, output_path="graph.png"):
    strong_edges = [(u, v) for u, v, d in G.edges(data=True)]
    
    G_vis = nx.Graph()
    G_vis.add_edges_from(strong_edges)
    G_vis.add_nodes_from(G.nodes())

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G_vis, k=0.15, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(G_vis, pos, node_size=60, node_color='skyblue', alpha=0.9, edgecolors='white')
    nx.draw_networkx_edges(G_vis, pos, alpha=0.3, edge_color='red')
    nx.draw_networkx_labels(G_vis, pos, labels={node: str(node) for node in G_vis.nodes()}, font_size=7)
    
    plt.title("Semantic Graph", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def build_and_visualize_semantic_graph(
    npy_path,
    time_steps_per_day,
    output_path="topo_input.npy",
    graph_path="graph.png",
    dimensions=32,
    top_k=10,
    seed=42,
):
    raw_data = np.load(npy_path)
    train_size = int(0.8* raw_data.shape[0])
    train_raw = raw_data[: train_size]
    

    T,C,H,W = train_raw.shape
    num_locations = H * W
    demand_data = train_raw[:,0,:,:]

    steps_per_week = 7 * time_steps_per_day
    num_weeks = T // steps_per_week

    demand_data = demand_data[:num_weeks * steps_per_week,:,:]
    weekly_avg = demand_data.reshape(num_weeks,steps_per_week,H,W).mean(axis = 0)

    weekly_patterns = weekly_avg.reshape(steps_per_week, num_locations).T
    pattern_mean = weekly_patterns.mean(axis=1, keepdims=True)
    pattern_std = weekly_patterns.std(axis=1, keepdims=True)
    weekly_patterns = (weekly_patterns - pattern_mean) / (pattern_std + 1e-7)

    G = nx.Graph()

    for i in range(num_locations):
        G.add_node(i)
    distances = []
    pairs = []
    for i in range(num_locations):
        for j in range(i+1,num_locations):
            dist = dtw(weekly_patterns[i],weekly_patterns[j])
            distances.append(dist)
            pairs.append((i,j))

    sigma = np.std(distances) if np.std(distances) > 0 else 1.0
    neighbors = {i: [] for i in range(num_locations)}
    for idx, (i, j) in enumerate(pairs):
        dist = distances[idx]
        weight = np.exp(- (dist ** 2) / (sigma ** 2))
        neighbors[i].append((j, weight))
        neighbors[j].append((i, weight))

    for i, weighted_neighbors in neighbors.items():
        for j, weight in sorted(weighted_neighbors, key=lambda item: item[1], reverse=True)[:top_k]:
            G.add_edge(i, j, weight=weight)

    visualize_semantic_graph(G, output_path=graph_path)

    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=10,
        num_walks=50,
        workers=2,
        weight_key='weight',
        seed=seed,
    )
    model = node2vec.fit(window=5, min_count=1, batch_words=4, seed=seed)
    embeddings = np.array([model.wv[str(i)] for i in range(num_locations)], dtype=np.float32)

    np.save(output_path,embeddings)
    print(f"Finish! nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, embeddings={embeddings.shape}")

if __name__ == '__main__':
    DATA_DIR = Path(__file__).resolve().parents[2] / 'data' / '2024'
    NPY_PATH = DATA_DIR / 'taxi_volume_4d_tensor.npy'
    OUTPUT_PATH = DATA_DIR / 'topo_input.npy'
    GRAPH_PATH = DATA_DIR / 'semantic_graph.png'

    TIME_STEPS_PER_DAY = 48

    build_and_visualize_semantic_graph(NPY_PATH, TIME_STEPS_PER_DAY, OUTPUT_PATH, GRAPH_PATH)
