import func

vital_page_1 = func.pwb.Page(func.site, 'Wikipédia:Articles vitaux/Niveau 1')
print(type(vital_page_1))
parsed_vital_page_1 = func.mwp.parse(vital_page_1.text, skip_style_tags=True)
text_1 = func.extract_text(parsed_vital_page_1)
index_1 = text_1.index('Les dix articles')
pages_1 = text_1[index_1 + len('Les dix articles'):].strip().split('\n')
list_of_pages_1 = []
for i in range(10):
  list_of_pages_1.append(pages_1[i].strip())

func.save_pages_1(list_of_pages_1)

df_1, G_1 = func.extract_pages_1(list_of_pages_1)

func.pickle.dump(df_1, open('data/l1_data.pkl', 'wb'))
func.pickle.dump(G_1, open('data/l1_graph.pkl', 'wb'))
