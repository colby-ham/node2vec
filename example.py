import networkx as nx
from node2vec import Node2Vec
import sys

# FILES
#EMBEDDING_FILENAME = './embeddings.emb'
#EMBEDDING_MODEL_FILENAME = './embeddings.model'
EDGELIST_FILENAME = sys.argv[1]
EMBEDDING_FILENAME = sys.argv[2]
EMBEDDING_MODEL_FILENAME = sys.argv[3]
# NOT using this option for temp folder yet
#EMBEDDING_TEMP_FOLDER_DIRNAME = sys.argv[4]
print(f"EDGELIST_FILENAME: {EDGELIST_FILENAME}")
print(f"EMBEDDING_FILENAME: {EMBEDDING_FILENAME}")
print(f"EMBEDDING_MODEL_FILENAME: {EMBEDDING_MODEL_FILENAME}")
#print(f"EMBEDDING_TEMP_FOLDER_DIRNAME {EMBEDDING_TEMP_FOLDER_DIRNAME}")


print("Read graph")
G = nx.read_edgelist(EDGELIST_FILENAME, nodetype=int, create_using=nx.DiGraph())
#for edge in G.edges():
#			G[edge[0]][edge[1]]['weight'] = 1

#print("Create a graph")
#graph = nx.fast_gnp_random_graph(n=100, p=0.5)

print("Precompute probabilities and generate walks")
node2vec = Node2Vec(G, dimensions=128, walk_length=80, num_walks=10, workers=8) #, temp_folder=EMBEDDING_TEMP_FOLDER_DIRNAME)

## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
# Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
#node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")

print("Embed")
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

#print("Look for most similar nodes")
#model.wv.most_similar('2')  # Output node names are always strings

print("Save embeddings for later use")
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

print("Save model for later use")
model.save(EMBEDDING_MODEL_FILENAME)
