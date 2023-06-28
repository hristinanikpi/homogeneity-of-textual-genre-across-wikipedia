import func 

df_2 = func.pickle.load(open('data/l2_data_wsw.pkl', 'rb'))

scores_node2vec_2 = func.pickle.load(open('results/l2_network_node2vec.pkl', 'rb'))
scores_simrank_2 = func.pickle.load(open('results/l2_network_simrank.pkl', 'rb'))
scores_panther_2 = func.pickle.load(open('results/l2_network_panther.pkl', 'rb'))

scores_nodes_2 = func.pickle.load(open('results_wsw/l2_doc2vec_nodes.pkl', 'rb'))
scores_nodes_with_words_2 = func.pickle.load(open('results_wsw/l2_doc2vec_nodes_with_words.pkl', 'rb'))
scores_words_2 = func.pickle.load(open('results_wsw/l2_doc2vec_words.pkl', 'rb'))
scores_parts_of_speech_2 = func.pickle.load(open('results_wsw/l2_doc2vec_parts_of_speech.pkl', 'rb'))

fig, ((ax1, ax2), (ax3, ax4)) = func.plt.subplots(2, 2, figsize=(14, 10), constrained_layout = True)
a, b = func.np.polyfit(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_2[func.np.triu_indices(df_2.shape[0], k = 1)], 1)
ax1.plot(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], a*scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)]+b, color='red')
ax1.scatter(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_2[func.np.triu_indices(df_2.shape[0], k = 1)], c='r', s=func.np.ones(len(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)])))
x = func.scipy.stats.spearmanr(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_2[func.np.triu_indices(df_2.shape[0], k = 1)])
text = 'correlation: ' + str(x.correlation) + '\n' + 'pvalue: ' + str(x.pvalue)
ax1.text(0.1, 0.1, text)
a, b = func.np.polyfit(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_with_words_2[func.np.triu_indices(df_2.shape[0], k = 1)], 1)
ax2.plot(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], a*scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)]+b, color='red')
ax2.scatter(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_with_words_2[func.np.triu_indices(df_2.shape[0], k = 1)], c='r', s=func.np.ones(len(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)])))
x = func.scipy.stats.spearmanr(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_with_words_2[func.np.triu_indices(df_2.shape[0], k = 1)])
text = 'correlation: ' + str(x.correlation) + '\n' + 'pvalue: ' + str(x.pvalue)
ax2.text(0.1, 0.83, text)    
a, b = func.np.polyfit(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_words_2[func.np.triu_indices(df_2.shape[0], k = 1)], 1)
ax3.plot(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], a*scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)]+b, color='red')
ax3.scatter(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_words_2[func.np.triu_indices(df_2.shape[0], k = 1)], c='r', s=func.np.ones(len(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)])))
x = func.scipy.stats.spearmanr(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_words_2[func.np.triu_indices(df_2.shape[0], k = 1)])
text = 'correlation: ' + str(x.correlation) + '\n' + 'pvalue: ' + str(x.pvalue)
ax3.text(0.1, 0.68, text)
a, b = func.np.polyfit(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_parts_of_speech_2[func.np.triu_indices(df_2.shape[0], k = 1)], 1)
ax4.plot(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], a*scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)]+b, color='red')
ax4.scatter(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_parts_of_speech_2[func.np.triu_indices(df_2.shape[0], k = 1)], c='r', s=func.np.ones(len(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)])))
x = func.scipy.stats.spearmanr(scores_node2vec_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_parts_of_speech_2[func.np.triu_indices(df_2.shape[0], k = 1)])
text = 'correlation: ' + str(x.correlation) + '\n' + 'pvalue: ' + str(x.pvalue)
ax4.text(0.1, 0.85, text)
fig.suptitle('Correlations between Node2Vec and Doc2Vec similarities on Level 2')
ax1.set_title('Node2Vec and Doc2Vec (nodes)')
ax2.set_title('Node2Vec and Doc2Vec (nodes with words)')
ax3.set_title('Node2Vec and Doc2Vec (words)')
ax4.set_title('Node2Vec and Doc2Vec (parts_of_speech)')
func.plt.savefig('figures_wsw/l2_correlations_part1.png')
func.plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = func.plt.subplots(2, 2, figsize=(14, 10), constrained_layout = True)
a, b = func.np.polyfit(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_2[func.np.triu_indices(df_2.shape[0], k = 1)], 1)
ax1.plot(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], a*scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)]+b, color='red')
ax1.scatter(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_2[func.np.triu_indices(df_2.shape[0], k = 1)], c='r', s=func.np.ones(len(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)])))
x = func.scipy.stats.spearmanr(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_2[func.np.triu_indices(df_2.shape[0], k = 1)])
text = 'correlation: ' + str(x.correlation) + '\n' + 'pvalue: ' + str(x.pvalue)
ax1.text(0.2, 0.1, text)
a, b = func.np.polyfit(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_with_words_2[func.np.triu_indices(df_2.shape[0], k = 1)], 1)
ax2.plot(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], a*scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)]+b, color='red')
ax2.scatter(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_with_words_2[func.np.triu_indices(df_2.shape[0], k = 1)], c='r', s=func.np.ones(len(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)])))
x = func.scipy.stats.spearmanr(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_with_words_2[func.np.triu_indices(df_2.shape[0], k = 1)])
text = 'correlation: ' + str(x.correlation) + '\n' + 'pvalue: ' + str(x.pvalue)
ax2.text(0.2, 0.83, text)
a, b = func.np.polyfit(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_words_2[func.np.triu_indices(df_2.shape[0], k = 1)], 1)
ax3.plot(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], a*scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)]+b, color='red')
ax3.scatter(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_words_2[func.np.triu_indices(df_2.shape[0], k = 1)], c='r', s=func.np.ones(len(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)])))
x = func.scipy.stats.spearmanr(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_words_2[func.np.triu_indices(df_2.shape[0], k = 1)])
text = 'correlation: ' + str(x.correlation) + '\n' + 'pvalue: ' + str(x.pvalue)
ax3.text(0.2, 0.68, text)
a, b = func.np.polyfit(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_parts_of_speech_2[func.np.triu_indices(df_2.shape[0], k = 1)], 1)
ax4.plot(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], a*scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)]+b, color='red')
ax4.scatter(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_parts_of_speech_2[func.np.triu_indices(df_2.shape[0], k = 1)], c='r', s=func.np.ones(len(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)])))
x = func.scipy.stats.spearmanr(scores_simrank_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_parts_of_speech_2[func.np.triu_indices(df_2.shape[0], k = 1)])
text = 'correlation: ' + str(x.correlation) + '\n' + 'pvalue: ' + str(x.pvalue)
ax4.text(0.2, 0.85, text)
fig.suptitle('Correlations between Simrank and Doc2Vec similarities on Level 2')
ax1.set_title('Simrank and Doc2Vec (nodes)')
ax2.set_title('Simrank and Doc2Vec (nodes with words)')
ax3.set_title('Simrank and Doc2Vec (words)')
ax4.set_title('Simrank and Doc2Vec (parts_of_speech)')
func.plt.savefig('figures_wsw/l2_correlations_part2.png')
func.plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = func.plt.subplots(2, 2, figsize=(14, 10), constrained_layout = True)
a, b = func.np.polyfit(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_2[func.np.triu_indices(df_2.shape[0], k = 1)], 1)
ax1.plot(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], a*scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)]+b, color='red')
ax1.scatter(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_2[func.np.triu_indices(df_2.shape[0], k = 1)], c='r', s=func.np.ones(len(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)])))
x = func.scipy.stats.spearmanr(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_2[func.np.triu_indices(df_2.shape[0], k = 1)])
text = 'correlation: ' + str(x.correlation) + '\n' + 'pvalue: ' + str(x.pvalue)
ax1.text(0.002, 0.1, text)
a, b = func.np.polyfit(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_with_words_2[func.np.triu_indices(df_2.shape[0], k = 1)], 1)
ax2.plot(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], a*scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)]+b, color='red')
ax2.scatter(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_with_words_2[func.np.triu_indices(df_2.shape[0], k = 1)], c='r', s=func.np.ones(len(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)])))
x = func.scipy.stats.spearmanr(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_nodes_with_words_2[func.np.triu_indices(df_2.shape[0], k = 1)])
text = 'correlation: ' + str(x.correlation) + '\n' + 'pvalue: ' + str(x.pvalue)
ax2.text(0.002, 0.83, text)
a, b = func.np.polyfit(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_words_2[func.np.triu_indices(df_2.shape[0], k = 1)], 1)
ax3.plot(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], a*scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)]+b, color='red')
ax3.scatter(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_words_2[func.np.triu_indices(df_2.shape[0], k = 1)], c='r', s=func.np.ones(len(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)])))
x = func.scipy.stats.spearmanr(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_words_2[func.np.triu_indices(df_2.shape[0], k = 1)])
text = 'correlation: ' + str(x.correlation) + '\n' + 'pvalue: ' + str(x.pvalue)
ax3.text(0.002, 0.68, text)
a, b = func.np.polyfit(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_parts_of_speech_2[func.np.triu_indices(df_2.shape[0], k = 1)], 1)
ax4.plot(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], a*scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)]+b, color='red')
ax4.scatter(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_parts_of_speech_2[func.np.triu_indices(df_2.shape[0], k = 1)], c='r', s=func.np.ones(len(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)])))
x = func.scipy.stats.spearmanr(scores_panther_2[func.np.triu_indices(df_2.shape[0], k = 1)], scores_parts_of_speech_2[func.np.triu_indices(df_2.shape[0], k = 1)])
text = 'correlation: ' + str(x.correlation) + '\n' + 'pvalue: ' + str(x.pvalue)
ax4.text(0.002, 0.85, text)
fig.suptitle('Correlations between Panther and Doc2Vec similarities on Level 2')
ax1.set_title('Panther and Doc2Vec (nodes)')
ax2.set_title('Panther and Doc2Vec (nodes with words)')
ax3.set_title('Panther and Doc2Vec (words)')
ax4.set_title('Panther and Doc2Vec (parts_of_speech)')
func.plt.savefig('figures_wsw/l2_correlations_part3.png')
func.plt.show()
