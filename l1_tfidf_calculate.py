import func 

df_1 = func.pickle.load(open('data/l1_data_wsw.pkl', 'rb'))

docs_nodes = df_1['Nodes']
dictionary_nodes = func.Dictionary(docs_nodes)
bow_corpus_nodes = [dictionary_nodes.doc2bow(doc) for doc in docs_nodes]
tfidf_nodes = func.models.TfidfModel(bow_corpus_nodes)
corpus_tfidf_nodes = tfidf_nodes[bow_corpus_nodes]
corpus_nodes_dense = func.corpus2dense(tfidf_nodes[bow_corpus_nodes], num_terms=len(dictionary_nodes)).T
scores_tfidf_nodes = func.cosine_similarity(corpus_nodes_dense)
func.pickle.dump(scores_tfidf_nodes, open('results_wsw/l1_tfidf_nodes.pkl', 'wb'))

docs_nodes_with_words = df_1['Nodes with words']
dictionary_nodes_with_words = func.Dictionary(docs_nodes_with_words)
bow_corpus_nodes_with_words = [dictionary_nodes_with_words.doc2bow(doc) for doc in docs_nodes_with_words]
tfidf_nodes_with_words = func.models.TfidfModel(bow_corpus_nodes)
corpus_tfidf_nodes_with_words = tfidf_nodes_with_words[bow_corpus_nodes_with_words]
corpus_nodes_with_words_dense = func.corpus2dense(tfidf_nodes_with_words[bow_corpus_nodes_with_words], num_terms=len(dictionary_nodes)).T
scores_tfidf_nodes_with_words = func.cosine_similarity(corpus_nodes_with_words_dense)
func.pickle.dump(scores_tfidf_nodes_with_words, open('results_wsw/l1_tfidf_nodes_with_words.pkl', 'wb'))

docs_words = df_1['Words']
dictionary_words = func.Dictionary(docs_words)
bow_corpus_words = [dictionary_words.doc2bow(doc) for doc in docs_words]
tfidf_words = func.models.TfidfModel(bow_corpus_words)
corpus_tfidf_words = tfidf_words[bow_corpus_words]
corpus_words_dense = func.corpus2dense(tfidf_words[bow_corpus_words], num_terms=len(dictionary_words)).T
scores_tfidf_words = func.cosine_similarity(corpus_words_dense)
func.pickle.dump(scores_tfidf_words, open('results_wsw/l1_tfidf_words.pkl', 'wb'))

docs_parts_of_speech = df_1['Parts of speech']
dictionary_parts_of_speech = func.Dictionary(docs_parts_of_speech)
bow_corpus_parts_of_speech = [dictionary_parts_of_speech.doc2bow(doc) for doc in docs_parts_of_speech]
tfidf_parts_of_speech = func.models.TfidfModel(bow_corpus_parts_of_speech)
corpus_tfidf_parts_of_speech = tfidf_parts_of_speech[bow_corpus_parts_of_speech]
corpus_parts_of_speech_dense = func.corpus2dense(tfidf_parts_of_speech[bow_corpus_parts_of_speech], num_terms=len(dictionary_parts_of_speech)).T
scores_tfidf_parts_of_speech = func.cosine_similarity(corpus_parts_of_speech_dense)
func.pickle.dump(scores_tfidf_parts_of_speech, open('results_wsw/l1_tfidf_parts_of_speech.pkl', 'wb'))
