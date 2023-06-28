import func

df_1 = func.pickle.load(open('data/l1_data.pkl', 'rb'))

nodes_1 = list(func.tagged_documents(df_1['Nodes']))
model_nodes_1 = func.Doc2Vec(nodes_1, dm=0, vector_size=100, min_count=2, epochs=500)
scores_nodes_1 = func.np.array([[model_nodes_1.wv.n_similarity(df_1.iloc[i]['Nodes'], df_1.iloc[j]['Nodes']) for i in range(df_1.shape[0])] for j in range(df_1.shape[0])])

func.pickle.dump(scores_nodes_1, open('results_1/l1_doc2vec_nodes.pkl', 'wb'))

words_1 = list(func.tagged_documents(df_1['Words']))
model_words_1 = func.Doc2Vec(words_1, dm=0, vector_size=500, min_count=2, epochs=500)
scores_words_1 = func.np.array([[model_words_1.wv.n_similarity(df_1.iloc[i]['Words'], df_1.iloc[j]['Words']) for i in range(df_1.shape[0])] for j in range(df_1.shape[0])])

func.pickle.dump(scores_words_1, open('results_1/l1_doc2vec_words.pkl', 'wb'))

nodes_with_words_1 = list(func.tagged_documents(df_1['Nodes with words']))
model_nodes_with_words_1 = func.Doc2Vec(nodes_with_words_1, dm=0, vector_size=500, min_count=2, epochs=500)
scores_nodes_with_words_1 = func.np.array([[model_nodes_with_words_1.wv.n_similarity(df_1.iloc[i]['Nodes with words'], df_1.iloc[j]['Nodes with words']) for i in range(df_1.shape[0])] for j in range(df_1.shape[0])])

func.pickle.dump(scores_nodes_with_words_1, open('results_1/l1_doc2vec_nodes_with_words.pkl', 'wb'))

parts_of_speech_1 = list(func.tagged_documents(df_1['Parts of speech']))
model_parts_of_speech_1 = func.Doc2Vec(parts_of_speech_1, dm=0, vector_size=100, min_count=2, epochs=500)
scores_parts_of_speech_1 = func.np.array([[model_parts_of_speech_1.wv.n_similarity(df_1.iloc[i]['Parts of speech'], df_1.iloc[j]['Parts of speech']) for i in range(df_1.shape[0])] for j in range(df_1.shape[0])])

func.pickle.dump(scores_parts_of_speech_1, open('results_1/l1_doc2vec_parts_of_speech.pkl', 'wb'))
