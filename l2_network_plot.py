import func 

df_2 = func.pickle.load(open('data/l2_data.pkl', 'rb'))

scores_node2vec_2 = func.pickle.load(open('results/l2_network_node2vec.pkl', 'rb'))
scores_simrank_2 = func.pickle.load(open('results/l2_network_simrank.pkl', 'rb'))
scores_panther_2 = func.pickle.load(open('results/l2_network_panther.pkl', 'rb'))


fig, (ax1, ax2) = func.plt.subplots(1, 2, figsize=(63, 30), constrained_layout = True)
fig.suptitle('Network Node2Vec Similarities on Level 2')
func.sns.heatmap(scores_node2vec_2, ax=ax1, xticklabels=df_2['Title'], yticklabels=df_2['Title'], vmin=0.0, vmax=1.0, cmap='Reds')
func.sns.histplot(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], ax=ax2)
ax1.set_title('Node2Vec Similarity Matrix')
func.plt.savefig('figures/l2_network_part1.png')
func.plt.show()

fig, (ax1, ax2) = func.plt.subplots(1, 2, figsize=(63, 30), constrained_layout = True)
fig.suptitle('Network Simrank Similarities on Level 2')
func.sns.heatmap(scores_simrank_2, ax=ax1, xticklabels=df_2['Title'], yticklabels=df_2['Title'], vmin=0.0, vmax=1.0, cmap='Reds')
func.sns.histplot(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], ax=ax2)
ax1.set_title('Simrank Similarity Matrix')
func.plt.savefig('figures/l2_network_part2.png')
func.plt.show()

fig, (ax1, ax2) = func.plt.subplots(1, 2, figsize=(63, 30), constrained_layout = True)
fig.suptitle('Network Panther Similarities on Level 2')
func.sns.heatmap(scores_panther_2, ax=ax1, xticklabels=df_2['Title'], yticklabels=df_2['Title'], vmin=0.0, vmax=1.0, cmap='Reds')
func.sns.histplot(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], ax=ax2)
ax1.set_title('Panther Similarity Matrix')
func.plt.savefig('figures/l2_network_part3.png')
func.plt.show()

'''
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = func.plt.subplots(2, 3, figsize=(150, 80), constrained_layout = True)
fig.suptitle('Network Smilarities on Level 1')
func.sns.heatmap(scores_node2vec_2, ax=ax1, xticklabels=df_2['Title'], yticklabels=df_2['Title'], vmin=0.0, vmax=1.0, cmap='Reds')
func.sns.heatmap(scores_simrank_2, ax=ax2, xticklabels=df_2['Title'], yticklabels=df_2['Title'], vmin=0.0, vmax=1.0, cmap='Reds')
func.sns.heatmap(scores_panther_2, ax=ax3, xticklabels=df_2['Title'], yticklabels=df_2['Title'], vmin=0.0, vmax=1.0, cmap='Reds')
func.sns.histplot(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], ax=ax4)
func.sns.histplot(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], ax=ax5)
func.sns.histplot(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], ax=ax6)
ax1.set_title('Node2Vec Similarity Matrix')
ax2.set_title('Simrank Similarity Matrix')
ax3.set_title('Panther Similarity Matrix')
func.plt.show()
func.plt.savefig('figures/l2_network.png')
'''
