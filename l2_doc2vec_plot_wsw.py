import func 

df_2 = func.pickle.load(open('data/l2_data_wsw.pkl', 'rb'))

scores_nodes_2 = func.pickle.load(open('results_wsw/l2_doc2vec_nodes.pkl', 'rb'))
scores_nodes_with_words_2 = func.pickle.load(open('results_wsw/l2_doc2vec_nodes_with_words.pkl', 'rb'))
scores_words_2 = func.pickle.load(open('results_wsw/l2_doc2vec_words.pkl', 'rb'))
scores_parts_of_speech_2 = func.pickle.load(open('results_wsw/l2_doc2vec_parts_of_speech.pkl', 'rb'))

fig, (ax1, ax2) = func.plt.subplots(1, 2, figsize=(42, 20), constrained_layout = True)
fig.suptitle('Doc2Vec Nodes Similarities on Level 2')
func.sns.heatmap(scores_nodes_2, ax=ax1, xticklabels=df_2['Title'], yticklabels=df_2['Title'], vmin=0.0, vmax=1.0, cmap='Reds')
func.sns.histplot(scores_nodes_2[func.np.triu_indices(df_2.shape[0], k = 1)], ax=ax2)
ax1.set_title('Nodes Similarity Matrix')
func.plt.savefig('figures_wsw/l2_doc2vec_nodes.png')
func.plt.show()

fig, (ax1, ax2) = func.plt.subplots(1, 2, figsize=(42, 20), constrained_layout = True)
fig.suptitle('Doc2Vec Nodes with Words Similarities on Level 2')
func.sns.heatmap(scores_nodes_with_words_2, ax=ax1, xticklabels=df_2['Title'], yticklabels=df_2['Title'], vmin=0.0, vmax=1.0, cmap='Reds')
func.sns.histplot(scores_nodes_with_words_2[func.np.triu_indices(df_2.shape[0], k = 1)], ax=ax2)
ax1.set_title('Nodes with Words Similarity Matrix')
func.plt.savefig('figures_wsw/l2_doc2vec_nodes_with_words.png')
func.plt.show()

fig, (ax1, ax2) = func.plt.subplots(1, 2, figsize=(42, 20), constrained_layout = True)
fig.suptitle('Doc2Vec Words Similarities on Level 2')
func.sns.heatmap(scores_words_2, ax=ax1, xticklabels=df_2['Title'], yticklabels=df_2['Title'], vmin=0.0, vmax=1.0, cmap='Reds')
func.sns.histplot(scores_words_2[func.np.triu_indices(df_2.shape[0], k = 1)], ax=ax2)
ax1.set_title('Words Similarity Matrix')
func.plt.savefig('figures_wsw/l2_doc2vec_words.png')
func.plt.show()

fig, (ax1, ax2) = func.plt.subplots(1, 2, figsize=(42, 20), constrained_layout = True)
fig.suptitle('Doc2Vec Parts of Speech Similarities on Level 2')
func.sns.heatmap(scores_parts_of_speech_2, ax=ax1, xticklabels=df_2['Title'], yticklabels=df_2['Title'], vmin=0.0, vmax=1.0, cmap='Reds')
func.sns.histplot(scores_parts_of_speech_2[func.np.triu_indices(df_2.shape[0], k = 1)], ax=ax2)
ax1.set_title('Parts of Speech Similarity Matrix')
func.plt.savefig('figures_wsw/l2_doc2vec_parts_of_speech.png')
func.plt.show()
