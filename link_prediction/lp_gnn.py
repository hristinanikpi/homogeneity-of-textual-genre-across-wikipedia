import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pickle 
import random 
import networkx as nx 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# get data
df = pickle.load(open('/projects/LaboratoireICAR/MACDIT/hristina/data/l2_data_wsw.pkl', 'rb'))
G = pickle.load(open('/projects/LaboratoireICAR/MACDIT/hristina/data/l2_graph.pkl', 'rb'))
features = pickle.load(open('/projects/LaboratoireICAR/MACDIT/hristina/features/l2_features.pkl', 'rb'))

node_features = np.array([features.iloc[np.where(df["Title"] == str(node))].to_numpy() for node in G.nodes()]).reshape(features.shape[0], features.shape[1])
scaler = MinMaxScaler()
#scaler = StandardScaler()
node_features = scaler.fit_transform(node_features)

# add self loops
for node in G.nodes():
    if (node, node) not in G.edges: 
        G.add_edge(node, node)

adjacency_matrix = nx.to_numpy_array(G)
# information about the node itself should be more relevant than the information about its neighbors
adjacency_matrix = adjacency_matrix + np.eye(adjacency_matrix.shape[0], adjacency_matrix.shape[1])

# matrix that contains both node features and information about neighbors 
product = np.dot(adjacency_matrix, node_features)

input_dim = 2*features.shape[1]
hidden_dim1 = 128
hidden_dim2 = 64
hidden_dim3 = 32
hidden_dim4 = 16
#hidden_dim5 = 8
output_dim = 1

pairs = []
labels = []
titles = []


for i in range(adjacency_matrix.shape[0]):
    for j in range(adjacency_matrix.shape[0]):
        if adjacency_matrix[i][j] == 1: 
            pairs.append(np.concatenate((product[i], product[j])))
            titles.append((df["Title"][i], df["Title"][j]))
            labels.append(1)



num_pos = len(labels)

pairs_neg = []
labels_neg = []
titles_neg = []

'''
# extracting negative edges without randomization 
for i in range(adjacency_matrix.shape[0]):
    if len(pairs_neg) >  num_pos:
        break
    for j in range(adjacency_matrix.shape[0]):
        if len(pairs_neg) >  num_pos:
            break
        if adjacency_matrix[i][j] == 0: 
            pairs_neg.append(np.concatenate((product[i], product[j])))
            titles_neg.append((df["Title"][i], df["Title"][j]))
            labels_neg.append(0)
'''

# extracting negative edges with randomization 
for i in range(adjacency_matrix.shape[0]):
    for j in range(adjacency_matrix.shape[0]):
        if adjacency_matrix[i][j] == 0: 
            pairs_neg.append(np.concatenate((product[i], product[j])))
            titles_neg.append((df["Title"][i], df["Title"][j]))
            labels_neg.append(0)

idx_neg = set(random.sample(list(range(len(pairs_neg))), num_pos))
pairs_neg = [n for i,n in enumerate(pairs_neg) if i in idx_neg]
labels_neg = [n for i,n in enumerate(labels_neg) if i in idx_neg]
titles_neg = [n for i,n in enumerate(titles_neg) if i in idx_neg]

pairs = pairs + pairs_neg
titles = titles + titles_neg
labels = labels + labels_neg 

pairs = np.array(pairs)
labels = np.array(labels)

# devide into train, test and validation set
pairs_train, pairs_test, labels_train, labels_test, titles_train, titles_test = train_test_split(pairs, labels, titles, test_size=0.2, random_state=42)
pairs_train, pairs_val, labels_train, labels_val, titles_train, titles_val = train_test_split(pairs_train, labels_train, titles_train, test_size=0.2, random_state=42)


# define the model
model = Sequential()
model.add(Dense(hidden_dim1, activation='relu', input_dim=input_dim))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(LeakyReLU(alpha=0.05))
model.add(Dense(hidden_dim2, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(LeakyReLU(alpha=0.05))
model.add(Dense(hidden_dim3, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.1))
#model.add(LeakyReLU(alpha=0.1))
model.add(Dense(hidden_dim4, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.1))
#model.add(LeakyReLU(alpha=0.1))
#model.add(Dense(hidden_dim5, activation='relu'))
model.add(Dense(output_dim, activation='sigmoid'))


#num_epochs=60
#learning_rate = 0.01
#decay_rate = learning_rate / num_epochs
#momentum = 0.8
#sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
optimizer = Adam(learning_rate=0.01)

# compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# train the model
num_epochs = 100
batch_size = 16
history_callback = tf.keras.callbacks.History()
model.fit(pairs_train, labels_train, epochs=num_epochs, batch_size=batch_size, validation_data=(pairs_val, labels_val), callbacks=[history_callback])


# Make predictions
predictions = model.predict(pairs_test)
#prediction = model.predict(pair_to_predict)
#print("Prdeiction for the edge between nodes", df['Title'][idx_i], "and", df['Title'][idx_j], ":", prediction[0][0])

threshold = 0.5
predicted_labels = []
num_right = 0 
num_wrong = 0
for i, prediction in enumerate(predictions):
    predicted_labels.append(prediction[0])
    if prediction[0] > threshold and G.has_edge(titles_test[i], titles_test[j]) is True:
        num_right += 1 
    elif prediction[0] > threshold and G.has_edge(titles_test[i], titles_test[j]) is False:
        num_wrong += 1
    elif prediction[0] < threshold and G.has_edge(titles_test[i], titles_test[j]) is False:
        num_right += 1 
    elif prediction[0] < threshold and G.has_edge(titles_test[i], titles_test[j]) is True:
        num_wrong += 1

print(num_right, ':', num_wrong)

# evaluate model performance
auc_roc_score = roc_auc_score(labels_test, predicted_labels)
print("AUC-ROC Score:", auc_roc_score)
ap_score = average_precision_score(labels_test, predicted_labels)
print("Average Precision (AP) Score:", ap_score)

# access the loss history
train_loss_history = history_callback.history['loss']
val_loss_history = history_callback.history['val_loss']
train_acc_history = history_callback.history['accuracy']
val_acc_history = history_callback.history['val_accuracy']

# plot the loss history
plt.figure()
plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss', c='b')
plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss', c='g')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/projects/LaboratoireICAR/MACDIT/hristina/plots/l2_lp_gnn_loss.png')
plt.show()

# plotting the accuracy history
plt.figure()
plt.plot(range(1, num_epochs + 1), train_acc_history, label='Training Accuracy', c='b')
plt.plot(range(1, num_epochs + 1), val_acc_history, label='Validation Accuracy', c='g')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('/projects/LaboratoireICAR/MACDIT/hristina/plots/l2_lp_gnn_acc.png')
plt.show()
