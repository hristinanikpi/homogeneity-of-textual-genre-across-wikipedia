import func

df_2 = func.pickle.load(open('data/l2_data.pkl', 'rb'))

dm = [1, 0]
vector_size = [100, 200, 400]
window = [1, 2, 3]
epochs = [100, 200, 400]
hs = [1, 0]
score = [0]
params = [{	'dm': item[0],
				'vector_size': item[1],
				'window': item[2],
				'epochs': item[3],
				'hs': item[4], 
				'score': item[5]} for item in list(func.itertools.product(*[dm, vector_size, window, epochs, hs, score]))]

 
nodes_2 = df_2['Nodes']
tagged_nodes_2 = list(func.tagged_documents(df_2['Nodes']))
evaluation_doc2vec_nodes_2 = func.evaluate_doc2vec_2(df_2, nodes_2, tagged_nodes_2, params, 5)
evaluation_doc2vec_nodes_2 = func.pd.DataFrame(evaluation_doc2vec_nodes_2)
evaluation_doc2vec_nodes_2 = evaluation_doc2vec_nodes_2.sort_values(by=['score'], ascending=False)
evaluation_doc2vec_nodes_2.reset_index(drop=True, inplace=True)
print(evaluation_doc2vec_nodes_2)

func.pickle.dump(evaluation_doc2vec_nodes_2, open('evaluations/l2_doc2vec_nodes.pkl', 'wb'))

'''
nodes_with_words_2 = df_2['Nodes with words']
tagged_nodes_with_words_2 = list(func.tagged_documents(df_2['Nodes with words']))
evaluation_doc2vec_nodes_with_words_2 = func.evaluate_doc2vec_2(df_2, nodes_with_words_2, tagged_nodes_with_words_2, params, 5)
evaluation_doc2vec_nodes_with_words_2 = func.pd.DataFrame(evaluation_doc2vec_nodes_with_words_2)
evaluation_doc2vec_nodes_with_words_2 = evaluation_doc2vec_nodes_with_words_2.sort_values(by=['score'], ascending=False)
evaluation_doc2vec_nodes_with_words_2.reset_index(drop=True, inplace=True)
print(evaluation_doc2vec_nodes_with_words_2)

func.pickle.dump(evaluation_doc2vec_nodes_with_words_2, open('evaluations/l2_doc2vec_nodes_with_words.pkl', 'wb'))


words_2 = df_2['Words']
tagged_words_2 = list(func.tagged_documents(df_2['Words']))
evaluation_doc2vec_words_2 = func.evaluate_doc2vec_2(df_2, words_2, tagged_words_2, params, 5)
evaluation_doc2vec_words_2 = func.pd.DataFrame(evaluation_doc2vec_words_2)
evaluation_doc2vec_words_2 = evaluation_doc2vec_words_2.sort_values(by=['score'], ascending=False)
evaluation_doc2vec_words_2.reset_index(drop=True, inplace=True)
print(evaluation_doc2vec_words_2)

func.pickle.dump(evaluation_doc2vec_words_2, open('evaluations/l2_doc2vec_words.pkl', 'wb'))


parts_of_speech_2 = df_2['Parts of speech']
tagged_parts_of_speech_2 = list(func.tagged_documents(df_2['Parts of speech']))
evaluation_doc2vec_parts_of_speech_2 = func.evaluate_doc2vec_2(df_2, parts_of_speech_2, tagged_parts_of_speech_2, params, 5)
evaluation_doc2vec_parts_of_speech_2 = func.pd.DataFrame(evaluation_doc2vec_parts_of_speech_2)
evaluation_doc2vec_parts_of_speech_2 = evaluation_doc2vec_parts_of_speech_2.sort_values(by=['score'], ascending=False)
evaluation_doc2vec_parts_of_speech_2.reset_index(drop=True, inplace=True)
print(evaluation_doc2vec_parts_of_speech_2)

func.pickle.dump(evaluation_doc2vec_parts_of_speech_2, open('evaluations/l2_doc2vec_parts_of_speech.pkl', 'wb'))
'''
