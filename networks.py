import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import os

# Assign group (not used in rewiring anymore but useful for metadata)
def opinion_group(opinion):
    if opinion in [-2, -1]:
        return -1
    elif opinion in [1, 2]:
        return 1
    else:
        return 0

# New helper: Get preferred nodes based on homophily and opinion
def get_preferred_candidates(G, u, homophily_level):
    opinion_u = G.nodes[u]['opinion']
    if opinion_u == 0:
        return []

    if homophily_level == -1.0:
        return [
            n for n in G.nodes
            if n != u and not G.has_edge(u, n)
            and G.nodes[n]['opinion'] == -opinion_u  # exact opposite
        ]
    elif homophily_level == 1.0:
        return [
            n for n in G.nodes
            if n != u and not G.has_edge(u, n)
            and G.nodes[n]['opinion'] == opinion_u  # exact same
        ]
    elif homophily_level >= 0:
        return [
            n for n in G.nodes
            if n != u and not G.has_edge(u, n)
            and G.nodes[n]['opinion'] * opinion_u > 0  # same sign
        ]
    else:
        return [
            n for n in G.nodes
            if n != u and not G.has_edge(u, n)
            and G.nodes[n]['opinion'] * opinion_u < 0  # opposite sign
        ]

def create_homophilic_network(nodes_df, m=2, homophily_level=0.0, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    n = len(nodes_df)
    # G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    G = nx.empty_graph(n=n)

    for _, row in nodes_df.iterrows():
        G.nodes[row['ID']]['name'] = row['Name']
        G.nodes[row['ID']]['opinion'] = row['Opinion']
        G.nodes[row['ID']]['group'] = opinion_group(row['Opinion'])

    # for u, v in list(G.edges()):
    #     opinion_u = G.nodes[u]['opinion']
    #     opinion_v = G.nodes[v]['opinion']

    #     # Determine if they are from same "side" of opinion
    #     same_sign = np.sign(opinion_u) == np.sign(opinion_v) and opinion_u != 0 and opinion_v != 0

    #     if homophily_level >= 0:
    #         prob_keep = homophily_level * int(same_sign) + (1 - homophily_level)
    #     else:
    #         prob_keep = -homophily_level * int(not same_sign) + (1 + homophily_level)

    #     if random.random() > prob_keep:
    #         G.remove_edge(u, v)
    #         candidates = get_preferred_candidates(G, u, homophily_level)
    #         if candidates:
    #             new_v = random.choice(candidates)
    #             G.add_edge(u, new_v)
    if homophily_level == 1.0:
        for i in range(n):
            for j in range(i+1, n):
                if nodes_df.loc[i, 'Opinion'] == nodes_df.loc[j, 'Opinion']:
                    G.add_edge(i, j)
    elif homophily_level == 0.0:
        for i in range(n):
            if nodes_df.loc[i, 'Opinion'] == 0:
                positive_nodes = nodes_df[nodes_df['Opinion'] > 0].index.tolist()
                negative_nodes = nodes_df[nodes_df['Opinion'] < 0].index.tolist()
                chosen_positive = random.sample(positive_nodes, min(11, len(positive_nodes)))
                chosen_negative = random.sample(negative_nodes, min(11, len(negative_nodes)))
                for j in chosen_positive + chosen_negative:
                    G.add_edge(i, j)
            else:
                for j in range(i+1, n):
                    if nodes_df.loc[i, 'Opinion'] == -nodes_df.loc[j, 'Opinion']:
                        G.add_edge(i, j)
    else:
        #TODO
        print("Homophily levels other than 0.0 and 1.0 are not implemented yet.")
    return G


df = pd.read_csv('persona_5point.csv')
nodes = df[['Name', 'Opinion']].copy()
nodes['ID'] = range(len(nodes))
#homophily_levels = [-1.0, -0.75, -0.5, -0.25,  0.0, 0.25, 0.5, 0.75, 1.0]
homophily_levels = [0.0, 0.5, 1.0]
networks = {h: create_homophilic_network(nodes, homophily_level=h) for h in homophily_levels}

output_dir = "network_plots"
os.makedirs(output_dir, exist_ok=True)

# Plot and export each network
for h, G in networks.items():
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)

    degrees = [G.degree(n) * 20 for n in G.nodes()]
    opinions = [G.nodes[n]['opinion'] for n in G.nodes()]
    cmap = plt.cm.RdYlBu

    nx.draw_networkx_nodes(
        G, pos,
        node_color=opinions,
        node_size=degrees,
        cmap=cmap,
        ax=ax
    )

    ax.set_title(f'Homophily = {h}', fontsize=14)

    # Zoom for nicer plots
    if h == 0:
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-1, 1)
    else:
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)

    ax.axis('off')

    # Save plot
    plt.savefig(f"{output_dir}/homophily_{h}_opinion_colored.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save edge list
    edges_data = []
    for u, v in G.edges():
        edges_data.append({
            'source': u,
            'target': v,
            'source_name': G.nodes[u]['name'],
            'target_name': G.nodes[v]['name'],
            'source_opinion': G.nodes[u]['opinion'],
            'target_opinion': G.nodes[v]['opinion'],
            'source_group': G.nodes[u]['group'],
            'target_group': G.nodes[v]['group']
        })
    edge_df = pd.DataFrame(edges_data)
    edge_df.to_csv(f"{output_dir}/homophily_{h}_edges.csv", index=False)
