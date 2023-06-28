import func 

df_1 = func.pickle.load(open('data/l1_data.pkl', 'rb'))
G_1 = func.pickle.load(open('data/l1_graph.pkl', 'rb')) 

node2vec = func.Node2Vec(G_1, dimensions=50, walk_length=20, num_walks=100)
model = node2vec.fit(window=10, min_count=1)
scores_node2vec_1 = func.np.zeros((df_1.shape[0], df_1.shape[0]))
for i in range(df_1.shape[0]):
  similar_1 = dict(model.wv.most_similar(df_1['Title'][i]))
  for j in range(df_1.shape[0]):
    if df_1['Title'][j] == df_1['Title'][i]:
      scores_node2vec_1[i][j] = 1.0 
    elif df_1['Title'][j] in similar_1.keys():
      scores_node2vec_1[i][j] = similar_1[df_1['Title'][j]]

func.pickle.dump(scores_node2vec_1, open('results/l1_network_node2vec.pkl', 'wb'))

simrank_1 = func.nx.simrank_similarity(G_1)
scores_simrank_1 = func.np.array([[v for v in values.values()] for values in simrank_1.values()])

func.pickle.dump(scores_simrank_1, open('results/l1_network_simrank.pkl', 'wb'))

scores_panther_1 = func.np.zeros((df_1.shape[0], df_1.shape[0]))
for i in range(df_1.shape[0]):
  panther_1 = func.nx.panther_similarity(G_1, df_1['Title'][i])
  for j in range(df_1.shape[0]):
    if df_1['Title'][j] == df_1['Title'][i]:
      scores_panther_1[i][j] = 1.0 
    elif df_1['Title'][j] in panther_1.keys():
      scores_panther_1[i][j] = panther_1[df_1["Title"][j]]
      
func.pickle.dump(scores_panther_1, open('results/l1_network_panther.pkl', 'wb'))
