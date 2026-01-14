import numpy as np
import pandas as pd
import networkx as nx
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

# Simulate grid graph (nodes: substations/meters, edges: transmission lines)
def generate_grid_graph(num_nodes=50, edge_prob=0.08):
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    for n in G.nodes:
        G.nodes[n]['load'] = np.random.uniform(10, 100)  # kWh
        G.nodes[n]['weather'] = np.random.uniform(0, 1)  # normalized risk
        G.nodes[n]['historical_outage'] = np.random.uniform(0, 1)
    return G

# Convert graph to PyTorch tensors
def graph_to_tensors(G):
    node_features = []
    for n in G.nodes:
        node_features.append([
            G.nodes[n]['load'],
            G.nodes[n]['weather'],
            G.nodes[n]['historical_outage']
        ])
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    return node_features, edge_index

# Simple GNN for grid risk prediction
class GridGNN(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x, edge_index):
        h = F.relu(self.fc1(x))
        # Aggregate neighbor features (mean)
        for i in range(x.size(0)):
            neighbors = (edge_index[0] == i).nonzero(as_tuple=True)[0]
            if len(neighbors) > 0:
                h[i] += h[edge_index[1][neighbors]].mean(dim=0)
        out = torch.sigmoid(self.fc2(h))
        return out.squeeze()

def predict_grid_risk(G):
    node_features, edge_index = graph_to_tensors(G)
    model = GridGNN(in_features=3, hidden_dim=8)
    with torch.no_grad():
        risk_scores = model(node_features, edge_index).numpy()
    return risk_scores

def get_grid_alerts(G, risk_scores, threshold=0.7):
    alerts = []
    for i, score in enumerate(risk_scores):
        if score > threshold:
            alerts.append(f"Node {i}: High blackout risk ({score:.2f})")
    return alerts

if __name__ == "__main__":
    G = generate_grid_graph()
    risk_scores = predict_grid_risk(G)
    alerts = get_grid_alerts(G, risk_scores)
    print("Grid Risk Scores:", risk_scores)
    print("Alerts:", alerts)
