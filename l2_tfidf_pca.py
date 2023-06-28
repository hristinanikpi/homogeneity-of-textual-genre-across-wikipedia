import func 

df_2 = func.pickle.load(open('data/l2_data_wsw.pkl', 'rb'))

pca = func.PCA(n_components=2)

docs_nodes = df_2['Nodes']
dictionary_nodes = func.Dictionary(docs_nodes)
bow_corpus_nodes = [dictionary_nodes.doc2bow(doc) for doc in docs_nodes]
tfidf_nodes = func.models.TfidfModel(bow_corpus_nodes)
corpus_nodes_dense = func.corpus2dense(tfidf_nodes[bow_corpus_nodes], num_terms=len(dictionary_nodes)).T
corpus_nodes_pca = pca.fit_transform(corpus_nodes_dense)
func.plt.scatter(corpus_nodes_pca[:,0], corpus_nodes_pca[:,1])
func.plt.xlabel('PCA Component 1')
func.plt.ylabel('PCA Component 2')
func.plt.title('TF-IDF PCA scatter plot for nodes on Level 2')
func.plt.savefig('figures_wsw/l2_tfidf_pca_nodes.png')
func.plt.show()

docs_nodes_with_words = df_2['Nodes with words']
dictionary_nodes_with_words = func.Dictionary(docs_nodes_with_words)
bow_corpus_nodes_with_words = [dictionary_nodes_with_words.doc2bow(doc) for doc in docs_nodes_with_words]
tfidf_nodes_with_words = func.models.TfidfModel(bow_corpus_nodes_with_words)
corpus_nodes_with_words_dense = func.corpus2dense(tfidf_nodes_with_words[bow_corpus_nodes_with_words], num_terms=len(dictionary_nodes_with_words)).T
corpus_nodes_with_words_pca = pca.fit_transform(corpus_nodes_with_words_dense)
func.plt.scatter(corpus_nodes_with_words_pca[:,0], corpus_nodes_with_words_pca[:,1])
func.plt.xlabel('PCA Component 1')
func.plt.ylabel('PCA Component 2')
func.plt.title('TF-IDF PCA scatter plot for nodes with words on Level 2')
func.plt.savefig('figures_wsw/l2_tfidf_pca_nodes_wit_words.png')
func.plt.show()

docs_words = df_2['Words']
dictionary_words = func.Dictionary(docs_words)
bow_corpus_words = [dictionary_words.doc2bow(doc) for doc in docs_words]
tfidf_words = func.models.TfidfModel(bow_corpus_words)
corpus_words_dense = func.corpus2dense(tfidf_words[bow_corpus_words], num_terms=len(dictionary_words)).T
corpus_words_pca = pca.fit_transform(corpus_words_dense)
func.plt.scatter(corpus_words_pca[:,0], corpus_words_pca[:,1])
func.plt.xlabel('PCA Component 1')
func.plt.ylabel('PCA Component 2')
func.plt.title('TF-IDF PCA scatter plot for words on Level 2')
func.plt.savefig('figures_wsw/l2_tfidf_pca_words.png')
func.plt.show()

docs_parts_of_speech = df_2['Parts of speech']
dictionary_parts_of_speech = func.Dictionary(docs_parts_of_speech)
bow_corpus_parts_of_speech = [dictionary_parts_of_speech.doc2bow(doc) for doc in docs_parts_of_speech]
tfidf_parts_of_speech = func.models.TfidfModel(bow_corpus_parts_of_speech)
corpus_parts_of_speech_dense = func.corpus2dense(tfidf_parts_of_speech[bow_corpus_parts_of_speech], num_terms=len(dictionary_parts_of_speech)).T
corpus_parts_of_speech_pca = pca.fit_transform(corpus_parts_of_speech_dense)
func.plt.scatter(corpus_parts_of_speech_pca[:,0], corpus_parts_of_speech_pca[:,1])
func.plt.xlabel('PCA Component 1')
func.plt.ylabel('PCA Component 2')
func.plt.title('TF-IDF PCA scatter plot for parts of speech on Level 2')
func.plt.savefig('figures_wsw/l2_tfidf_pca_parts_of_speech.png')
func.plt.show()
