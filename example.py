import networkx as nx
from node2vec import Node2Vec
import sys

# FILES
#EMBEDDING_FILENAME = './embeddings.emb'
#EMBEDDING_MODEL_FILENAME = './embeddings.model'
EMBEDDING_FILENAME = sys.argv[1]
EMBEDDING_MODEL_FILENAME = sys.argv[2]
print(f"EMBEDDING_FILENAME: ${EMBEDDING_FILENAME}")
print(f"EMBEDDING_MODEL_FILENAME: ${EMBEDDING_MODEL_FILENAME}")

print("Create a graph")
graph = nx.fast_gnp_random_graph(n=100, p=0.5)

print("Precompute probabilities and generate walks")
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)

## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
# Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
#node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")

print("Embed")
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

print("Look for most similar nodes")
model.wv.most_similar('2')  # Output node names are always strings

print("Save embeddings for later use")
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

print("Save model for later use")
model.save(EMBEDDING_MODEL_FILENAME)
