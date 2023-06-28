import func 

list_of_pages_2, list_of_categories_2 = func.extract_titles_2()

func.save_pages_2(list_of_pages_2)
    
df_2, G_2 = func.extract_pages_2(list_of_pages_2, list_of_categories_2)

func.pickle.dump(df_2, open('data/l2_data.pkl', 'wb'))
func.pickle.dump(G_2, open('data/l2_graph.pkl', 'wb'))
