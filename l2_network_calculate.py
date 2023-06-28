import func 

df_2 = func.pickle.load(open('data/l2_data.pkl', 'rb'))
G_2 = func.pickle.load(open('data/l2_graph.pkl', 'rb')) 

node2vec = func.Node2Vec(G_2, dimensions=50, walk_length=20, num_walks=100)
model = node2vec.fit(window=10, min_count=1)
scores_node2vec_2 = func.np.zeros((df_2.shape[0], df_2.shape[0]))
for i in range(df_2.shape[0]):
  similar_2 = dict(model.wv.most_similar(df_2['Title'][i]))
  for j in range(df_2.shape[0]):
    if df_2['Title'][j] == df_2['Title'][i]:
      scores_node2vec_2[i][j] = 1.0 
    elif df_2['Title'][j] in similar_2.keys():
      scores_node2vec_2[i][j] = similar_2[df_2['Title'][j]]

func.pickle.dump(scores_node2vec_2, open('results/l2_network_node2vec.pkl', 'wb'))

simrank_2 = func.nx.simrank_similarity(G_2)
scores_simrank_2 = func.np.array([[v for v in values.values()] for values in simrank_2.values()])

func.pickle.dump(scores_simrank_2, open('results/l2_network_simrank.pkl', 'wb'))

scores_panther_2 = func.np.zeros((df_2.shape[0], df_2.shape[0]))
for i in range(df_2.shape[0]):
  panther_2 = func.nx.panther_similarity(G_2, df_2['Title'][i])
  for j in range(df_2.shape[0]):
    if df_2['Title'][j] == df_2['Title'][i]:
      scores_panther_2[i][j] = 1.0 
    elif df_2['Title'][j] in panther_2.keys():
      scores_panther_2[i][j] = panther_2[df_2["Title"][j]]
      
func.pickle.dump(scores_panther_2, open('results/l2_network_panther.pkl', 'wb'))
