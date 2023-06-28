import networkx as nx
import numpy as np
import pickle
import random 
import math 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import seaborn as sns 


def common_neighbours(G, node1, node2):
    neighbors1 = set(G.neighbors(node1))
    neighbors2 = set(G.neighbors(node2))

    return len(neighbors1.intersection(neighbors2))

def jaccard_coeff(G, node1, node2):
    neighbors1 = set(G.neighbors(node1))
    neighbors2 = set(G.neighbors(node2))
    
    common_neighbors = neighbors1.intersection(neighbors2)
    union_neighbors = neighbors1.union(neighbors2)
    
    return len(common_neighbors) / len(union_neighbors)

def preferential_attachment(G, node1, node2):
    neighbors1 = set(G.neighbors(node1))
    neighbors2 = set(G.neighbors(node2))
    
    return len(neighbors1) * len(neighbors2)

def adamic_adar(G, node1, node2):
    common_neighbors = list(nx.common_neighbors(G, node1, node2))
    score = 0.0
    for neighbor in common_neighbors:
        degree = G.degree(neighbor)
        if degree > 1:
            score += 1.0 / math.log(degree)
    return score

def resource_allocation(G, node1, node2):
    common_neighbors = list(nx.common_neighbors(G, node1, node2))
    score = 0.0
    for neighbor in common_neighbors:
        degree = G.degree(neighbor)
        if degree > 1:
            score += 1.0 / degree
    return score

def link_prediction_with_features(G, df, features, heuristic, node1, node2):
    coeff = heuristic(G, node1, node2)
    feature_similarity = cosine_similarity(features.iloc[np.where(df["Title"] == str(node1))].to_numpy(), features.iloc[np.where(df["Title"] == str(node2))].to_numpy())
    combined_score = coeff * feature_similarity
    
    return combined_score

df_2 = pickle.load(open('/projects/LaboratoireICAR/MACDIT/hristina/data/l2_data_wsw.pkl', 'rb'))
G = pickle.load(open('/projects/LaboratoireICAR/MACDIT/hristina/data/l2_graph.pkl', 'rb'))
features = pickle.load(open('/projects/LaboratoireICAR/MACDIT/hristina/features/l2_features.pkl', 'rb'))

edges = list(G.edges)

non_edges_all = []
for node1 in G.nodes():
    for node2 in G.nodes():
        if node1 != node2 and not G.has_edge(node1, node2):
            non_edges_all.append((node1, node2))


inds_remove = set(random.sample(list(range(len(edges))), int(0.2*len(edges))))
inds_non = set(random.sample(list(range(len(non_edges_all))), int(0.2*len(edges))))
edges_to_remove = [n for i,n in enumerate(edges) if i in inds_remove]

non_edges = [n for i,n in enumerate(non_edges_all) if i in inds_non]


scores_removed = []
scores_removed_scaled = []
for edge in edges_to_remove:
    score = link_prediction_with_features(G, df_2, features, jaccard_coeff, edge[0], edge[1])
    score_scaled = 1 / (1 + math.exp(-score[0][0]))
    #print(edge[0], edge[1], score)
    scores_removed.append(score[0][0])
    scores_removed_scaled.append(score_scaled)

scores_removed.sort()
#scores_removed_scaled.sort()
print(scores_removed[0])
print(scores_removed[len(scores_removed)-1])


scores_non = []
scores_non_scaled = []
for edge in non_edges:
    score = link_prediction_with_features(G, df_2, features, jaccard_coeff, edge[0], edge[1])
    score_scaled = 1 / (1 + math.exp(-score[0][0]))
    #print(edge[0], edge[1], score)
    scores_non.append(score[0][0])
    scores_non_scaled.append(score_scaled)
scores_non.sort()
#scores_non_scaled.sort()
print(scores_non[0])
print(scores_non[len(scores_non)-1])

target_scores = np.concatenate((np.ones(len(scores_removed)), np.zeros(len(scores_non))))
print(target_scores)
predicted_scores = np.concatenate((np.array(scores_removed_scaled), np.array(scores_non_scaled)))
print(predicted_scores)
auc_roc_score = roc_auc_score(target_scores, predicted_scores)
print("AUC-ROC Score:", auc_roc_score)
ap_score = average_precision_score(target_scores, predicted_scores)
print("Average Precision (AP) Score:", ap_score)


plt.figure()
plt.plot(scores_removed, 'go', scores_non, 'ro')
plt.savefig('/projects/LaboratoireICAR/MACDIT/hristina/plots/l2_lp_pa.png')
plt.show()

plt.figure()
plt.plot(scores_removed_scaled, 'go', scores_non_scaled, 'ro')
plt.savefig('/projects/LaboratoireICAR/MACDIT/hristina/plots/l2_lp_pa_scaled.png')
plt.show()


fig, (ax1, ax2) = plt.subplots(1,2, constrained_layout = True)
sns.histplot(scores_removed, ax=ax1)
sns.histplot(scores_non, ax=ax2)
ax1.set_title('JC on existing edges')
ax2.set_title('JC on non-existing edges')
plt.savefig('/projects/LaboratoireICAR/MACDIT/hristina/plots/l2_lp_jc_histogram.png')
plt.show()



