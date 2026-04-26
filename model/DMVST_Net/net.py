import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from node2vec import Node2Vec

def visualize_semantic_graph(G, threshold=0.8):
    strong_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold]
    
    G_vis = nx.Graph()
    G_vis.add_edges_from(strong_edges)
    G_vis.add_nodes_from(G.nodes())

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G_vis, k=0.15, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(G_vis, pos, node_size=60, node_color='skyblue', alpha=0.9, edgecolors='white')
    nx.draw_networkx_edges(G_vis, pos, alpha=0.3, edge_color='red')
    
    plt.title(f"Semantic Graph (Threshold > {threshold})", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("graph.png", dpi=300)
    plt.show()

def build_and_visualize_semantic_graph(npy_path, time_steps_per_day,output_path="topo_input.npy"):
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

    G = nx.Graph()

    for i in range(num_locations):
        G.add_node(i)
    for i in range(num_locations):
        for j in range(i+1,num_locations):
            dist = dtw(weekly_patterns[i],weekly_patterns[j])
            weight = np.exp(-1.0 * dist)

            G.add_edge(i,j,weight=weight)
    visualize_semantic_graph(G, threshold=0.1)  

    node2vec = Node2Vec(G, dimensions=32, walk_length=10, num_walks=50, workers=2)
    model = node2vec.fit()
    embeddings = np.array([model.wv[str(i)] for i in range(num_locations)])

    np.save(output_path,embeddings)
    print("Finish")

if __name__ == '__main__':
    DATA_DIR = '/content/drive/MyDrive/mining_dataset/data/'
    NPY_PATH = DATA_DIR + 'taxi_volume_4d_tensor.npy'
    OUTPUT_PATH = DATA_DIR + 'topo_input.npy'

    TIME_STEPS_PER_DAY = 48

    build_and_visualize_semantic_graph(NPY_PATH, TIME_STEPS_PER_DAY, OUTPUT_PATH)