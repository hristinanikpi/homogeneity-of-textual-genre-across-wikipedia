from func import *

df_2 = pickle.load(open('data/l2_data_wsw.pkl', 'rb'))
#df_tmp = df_3.loc[df_3['Title'] == 'Ordinateur']

features = pickle.load(open('/projects/LaboratoireICAR/MACDIT/hristina/features/l2_features.pkl', 'rb'))
print(features.columns[0], features.iloc[95][0])
print(features.columns[1], features.iloc[95][1])
print(features.columns[2], features.iloc[95][2])

print(features.columns[16], features.iloc[95][16])
print(features.columns[17], features.iloc[95][17])
print(features.columns[18], features.iloc[95][18])

print(features.columns[51], features.iloc[95][51])
print(features.columns[52], features.iloc[95][52])
print(features.columns[53], features.iloc[95][63])

print(features.columns[70], features.iloc[95][70])
print(features.columns[71], features.iloc[95][71])

print(features.columns[121], features.iloc[95][121])
print(features.columns[122], features.iloc[95][122])
print(features.columns[123], features.iloc[95][123])

print(features.columns[120:140])
'''

G_2 = pickle.load(open('data/l2_graph.pkl', 'rb'))
def nudge(pos, x_shift, y_shift):
    return {n:(x + x_shift, y + y_shift) for n,(x,y) in pos.items()}

pos = nx.circular_layout(G_2)
pos_nodes = nudge(pos, 0.1, 0.1)                              
plt.figure(figsize=(8.2, 7))                    
nx.draw_networkx(G_2, pos=pos, with_labels=False, node_color='red', node_size=100, font_size=10)   # default nodes and edges
nx.draw_networkx_labels(G_2, pos=pos_nodes)         # nudged labels     # expand plot to fit labels
plt.show()

'''


#print(df_2['Title'][95])
#print(df_2['Nodes'][0])
#print(df_2['Words'][0])
#print(df_1['Nodes with words'][0])
#print(df_1['Parts of speech'][0])

