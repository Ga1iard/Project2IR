import faiss
import numpy as np

# Construye un índice vectorial FAISS 
def build_vector_index(embeddings):
    dimension = embeddings.shape[1]  # se obtiene la dimensión de los vectores
    print(dimension)

    # Se utiliza un índice plano con producto interno
    index = faiss.IndexFlatIP(dimension)

    # Se agregan los vectores al índice
    index.add(np.array(embeddings))

    return index

# Realiza una búsqueda de los K vectores más similares al embedding de consulta
def search_similar_vectors(query_emb, index, K):
    query_emb = np.asarray(query_emb, dtype='float32')

    # Asegura que el embedding de consulta tenga 2 dimensiones
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    elif query_emb.ndim != 2:
        raise ValueError(f"query_emb debe tener 1 o 2 dimensiones, pero tiene {query_emb.ndim}")

    # Retorna distancias y los índices de los K más cercanos
    D, I = index.search(query_emb, K)
    return D, I

# Guarda el índice en disco
def save_index(index, path="index_images.faiss"):
    faiss.write_index(index, path)

# Carga un índice desde disco
def load_index(path="index_images.faiss"):
    return faiss.read_index(path)
