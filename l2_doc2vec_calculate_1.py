import func

df_2 = func.pickle.load(open('data/l2_data.pkl', 'rb'))

nodes_2 = list(func.tagged_documents(df_2['Nodes']))
model_nodes_2 = func.Doc2Vec(nodes_2, dm=0, vector_size=100, min_count=2, epochs=500)
scores_nodes_2 = func.np.array([[model_nodes_2.wv.n_similarity(df_2.iloc[i]['Nodes'], df_2.iloc[j]['Nodes']) for i in range(df_2.shape[0])] for j in range(df_2.shape[0])])

func.pickle.dump(scores_nodes_2, open('results_1/l2_doc2vec_nodes.pkl', 'wb'))

nodes_with_words_2 = list(func.tagged_documents(df_2['Nodes with words']))
model_nodes_with_words_2 = func.Doc2Vec(nodes_with_words_2, dm=0, vector_size=500, min_count=2, epochs=500)
scores_nodes_with_words_2 = func.np.array([[model_nodes_with_words_2.wv.n_similarity(df_2.iloc[i]['Nodes with words'], df_2.iloc[j]['Nodes with words']) for i in range(df_2.shape[0])] for j in range(df_2.shape[0])])

func.pickle.dump(scores_nodes_with_words_2, open('results_1/l2_doc2vec_nodes_with_words.pkl', 'wb'))

words_2 = list(func.tagged_documents(df_2['Words']))
model_words_2 = func.Doc2Vec(words_2, dm=0, vector_size=500, min_count=2, epochs=500)
scores_words_2 = func.np.array([[model_words_2.wv.n_similarity(df_2.iloc[i]['Words'], df_2.iloc[j]['Words']) for i in range(df_2.shape[0])] for j in range(df_2.shape[0])])

func.pickle.dump(scores_words_2, open('results_1/l2_doc2vec_words.pkl', 'wb'))

parts_of_speech_2 = list(func.tagged_documents(df_2['Parts of speech']))
model_parts_of_speech_2 = func.Doc2Vec(parts_of_speech_2, dm=0, vector_size=100, min_count=2, epochs=500)
scores_parts_of_speech_2 = func.np.array([[model_parts_of_speech_2.wv.n_similarity(df_2.iloc[i]['Parts of speech'], df_2.iloc[j]['Parts of speech']) for i in range(df_2.shape[0])] for j in range(df_2.shape[0])])

func.pickle.dump(scores_parts_of_speech_2, open('results_1/l2_doc2vec_parts_of_speech.pkl', 'wb'))
