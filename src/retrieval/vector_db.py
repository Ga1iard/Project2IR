import faiss
import numpy as np

def build_vector_index(embeddings):
    dimension = embeddings.shape[1]
    print(dimension)

    # Distancia euclideana
    index = faiss.IndexFlatIP(dimension)  

    # Agregar los embeddings al índice
    index.add(np.array(embeddings))       

    return index


def search_similar_vectors(query_emb, index, K):
    query_emb = np.asarray(query_emb, dtype='float32')
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    elif query_emb.ndim != 2:
        raise ValueError(f"query_emb debe tener 1 o 2 dimensiones, pero tiene {query_emb.ndim}")

    D, I = index.search(query_emb, K)
    return D, I

def save_index(index, path="index_images.faiss"):
    faiss.write_index(index, path)

def load_index(path="index_images.faiss"):
    return faiss.read_index(path)