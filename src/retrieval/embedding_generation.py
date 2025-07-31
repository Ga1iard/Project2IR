from sentence_transformers import SentenceTransformer
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import torch

# Genera embeddings para una lista de textos usando un modelo CLIP compatible con SentenceTransformer
def generate_text_embeddings(texts: list[str], model_name='clip-ViT-B-32') -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

# Genera embeddings para todas las imágenes en un directorio, en batches, usando CLIP
def generate_image_embeddings(image_dir: str, model_name='clip-ViT-B-32', batch_size=32, resize_to=(224, 224)) -> tuple[np.ndarray, list[str]]:
    model = SentenceTransformer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Usando dispositivo: {device}")

    # Obtener rutas válidas de imágenes del directorio
    valid_exts = ('.jpg', '.jpeg', '.png')
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]

    all_embeddings = []

    # Procesar imágenes por lotes y codificarlas con el modelo
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Procesando en batches"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB').resize(resize_to)
                batch_images.append(img)
            except Exception as e:
                print(f"Error con imagen {path}: {e}")
        
        if batch_images:
            batch_embeds = model.encode(
                batch_images,
                convert_to_numpy=True,
                device=device,
                show_progress_bar=False
            )
            all_embeddings.append(batch_embeds)

    embeddings = np.vstack(all_embeddings)
    return embeddings, image_paths

# Genera embedding para una sola imagen individual
def generate_image_embedding(image_path: str, model_name='clip-ViT-B-32', resize_to=(224, 224)) -> np.ndarray:
    model = SentenceTransformer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    try:
        img = Image.open(image_path).convert('RGB').resize(resize_to)
    except Exception as e:
        raise ValueError(f"No se pudo abrir la imagen: {e}")

    # Codifica la imagen en un solo embedding
    embedding = model.encode([img], convert_to_numpy=True, device=device)[0]
    return embedding

# Combina embeddings de texto e imagen con un peso alpha, y normaliza el resultado
def combine_embeddings(txt_embeddings, img_embeddings, alpha=0.5):
    combined_embeddings = []
    for text_vec, image_vec in zip(txt_embeddings, img_embeddings):
        text_vec = np.array(text_vec)
        image_vec = np.array(image_vec)

        # Normalización individual de cada vector
        text_vec /= np.linalg.norm(text_vec)
        image_vec /= np.linalg.norm(image_vec)

        # Combinación ponderada de texto e imagen
        combined = alpha * text_vec + (1 - alpha) * image_vec
        combined /= np.linalg.norm(combined)  # Normalizar el embedding combinado

        combined_embeddings.append(combined)
    
    return combined_embeddings
