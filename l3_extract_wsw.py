import func 

list_of_pages_3, list_of_categories_3, list_of_subcategories_3 = func.extract_titles_3()

#func.save_pages_3(list_of_pages_3)

#G_3 = func.extract_graph_3(list_of_pages_3)
#func.pickle.dump(G_3, open('data/l3_graph.pkl', 'wb'))

'''
df = func.pickle.load(open('data/l3_data_tmp.pkl', 'rb'))
df_tmp = func.extract_pages_wsw_3(list_of_pages_3, list_of_categories_3, list_of_subcategories_3, 950, len(list_of_pages_3))
df_3 = func.pd.concat([df, df_tmp] , axis=0)
func.pickle.dump(df_3, open('data/l3_data_tmp.pkl', 'wb'))
print(df_3.shape)

'''
df_3 = func.pickle.load(open('data/l3_data_tmp.pkl', 'rb'))
G_3 = func.pickle.load(open('data/l3_graph.pkl', 'rb'))

for node in list(G_3.nodes):
	if G_3.degree[node] == 0:
		G_3.remove_node(node)
		df_3.drop(df_3[df_3['Title'] == node].index, inplace=True)
      
df_3.reset_index(drop=True, inplace=True)

func.pickle.dump(df_3, open('data/l3_data_wsw.pkl', 'wb'))
func.pickle.dump(G_3, open('data/l3_graph.pkl', 'wb'))

