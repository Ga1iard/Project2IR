import numpy as np
from src.retrieval.embedding_generation import generate_text_embeddings, generate_image_embedding
from src.retrieval.vector_db import search_similar_vectors

# Recuperación basada únicamente en texto
def retrieve_by_text(query_text, index, k):
    text_emb = emb_text_query(query_text)
    return search_similar_vectors(text_emb, index, k)

# Recuperación basada únicamente en imagen
def retrieve_by_image(query_image_path, index, k):
    image_emb = emb_image_query(query_image_path)
    return search_similar_vectors(image_emb, index, k)

# Recuperación combinando texto e imagen
def retrieve_by_text_and_image(query_text, query_image_path, index, k):
    text_emb = emb_text_query(query_text)
    image_emb = emb_image_query(query_image_path)

    # Normalización de ambos embeddings para mantener la escala
    text_emb = text_emb / np.linalg.norm(text_emb)
    image_emb = image_emb / np.linalg.norm(image_emb)

    # Ajuste del peso relativo entre texto e imagen según la longitud de la consulta textual
    if len(query_text.split()) <= 4:
        alpha = 0.3  # se favorece la imagen si el texto es muy corto
    else:
        alpha = 0.5  # peso equilibrado entre texto e imagen

    combined_emb = alpha * text_emb + (1 - alpha) * image_emb

    return search_similar_vectors(combined_emb, index, k)

# Función auxiliar para obtener embedding de texto desde una consulta
def emb_text_query(query):
    txt_embeddings = generate_text_embeddings([query])
    print(f"Texto: {query}")
    print(f"Shape del embedding: {txt_embeddings.shape}")  # Debería ser (1, 512)
    return txt_embeddings[0]

# Función auxiliar para obtener embedding de una imagen desde su ruta
def emb_image_query(query_path):
    img_embeddings = generate_image_embedding(query_path)
    return img_embeddings
