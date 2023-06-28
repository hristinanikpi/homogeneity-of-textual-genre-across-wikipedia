import func 

list_of_pages_4, list_of_categories_4 = func.extract_titles_4()

for i in range(len(list_of_pages_4)):
	if list_of_categories_4[i] == 'Histoire':
		print(list_of_pages_4[i])

#func.save_pages_4(list_of_pages_4)

#G_4 = func.extract_graph_4(list_of_pages_4)
#func.pickle.dump(G_4, open('data/l4_graph.pkl', 'wb'))

'''
df_nodes_3 = func.extract_nodes_3(list_of_pages_3, list_of_categories_3, list_of_subcategories_3)
func.pickle.dump(df_nodes_3, open('data/l3_data_nodes.pkl', 'wb'))

#df_words_3 = func.extract_words_3(list_of_pages_3)
#func.pickle.dump(df_words_3, open('data/l3_data_words.pkl', 'wb'))

#df_nodes_3 = func.pickle.load(open('data/l3_data_nodes.pkl', 'rb'))
#df_words_3 = func.pickle.load(open('data/l3_data_words.pkl', 'rb'))
#df_3 = func.pd.concat([df_nodes_3, df_words_3], axis=1)
#func.pickle.dump(df_3, open('data/l3_data.pkl', 'wb'))


#df = func.pickle.load(open('data/l3_data_tmp.pkl', 'rb'))
#df_tmp = func.extract_pages_3(list_of_pages_3, list_of_categories_3, list_of_subcategories_3, 950, len(list_of_pages_3))
#df_3 = func.pd.concat([df, df_tmp] , axis=0)
#func.pickle.dump(df_3, open('data/l3_data_tmp.pkl', 'wb'))

df_3 = func.pickle.load(open('data/l3_data_tmp.pkl', 'rb'))
G_3 = func.pickle.load(open('data/l3_graph.pkl', 'rb'))


for node in list(G_3.nodes):
	if G_3.degree[node] == 0:
		G_3.remove_node(node)
		df_3.drop(df_3[df_3['Title'] == node].index, inplace=True)
      
df_3.reset_index(drop=True, inplace=True)

func.pickle.dump(df_3, open('data/l3_data.pkl', 'wb'))
func.pickle.dump(G_3, open('data/l3_graph.pkl', 'wb'))
'''
