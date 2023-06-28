import func 

df_1 = func.pickle.load(open('data/l1_data.pkl', 'rb'))

scores_node2vec_1 = func.pickle.load(open('results/l1_network_node2vec.pkl', 'rb'))
scores_simrank_1 = func.pickle.load(open('results/l1_network_simrank.pkl', 'rb'))
scores_panther_1 = func.pickle.load(open('results/l1_network_panther.pkl', 'rb'))

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = func.plt.subplots(2, 3, figsize=(15, 8), constrained_layout = True)
fig.suptitle('Network Smilarities on Level 1')
func.sns.heatmap(scores_node2vec_1, ax=ax1, xticklabels=df_1['Title'], yticklabels=df_1['Title'], vmin=0.0, vmax=1.0, cmap='Reds')
func.sns.heatmap(scores_simrank_1, ax=ax2, xticklabels=df_1['Title'], yticklabels=df_1['Title'], vmin=0.0, vmax=1.0, cmap='Reds')
func.sns.heatmap(scores_panther_1, ax=ax3, xticklabels=df_1['Title'], yticklabels=df_1['Title'], vmin=0.0, vmax=1.0, cmap='Reds')
func.sns.histplot(scores_node2vec_1[func.np.triu_indices(df_1.shape[0], k = 1)], ax=ax4)
func.sns.histplot(scores_simrank_1[func.np.triu_indices(df_1.shape[0], k = 1)], ax=ax5)
func.sns.histplot(scores_panther_1[func.np.triu_indices(df_1.shape[0], k = 1)], ax=ax6)
ax1.set_title('Node2Vec Similarity Matrix')
ax2.set_title('Simrank Similarity Matrix')
ax3.set_title('Panther Similarity Matrix')
func.plt.savefig('figures/l1_network.png')
func.plt.show()

