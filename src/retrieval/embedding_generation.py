from sentence_transformers import SentenceTransformer
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import torch

def generate_text_embeddings(texts: list[str], model_name='clip-ViT-B-32') -> np.ndarray:
    """
    Genera embeddings para una lista de textos utilizando SentenceTransformer (CLIP).

    Parámetros:
        texts (list[str]): Lista de textos.
        model_name (str): Nombre del modelo en Hugging Face.

    Retorna:
        np.ndarray: Matriz de embeddings (n_docs x embedding_dim).
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


def generate_image_embeddings(image_dir: str, model_name='clip-ViT-B-32', batch_size=32, resize_to=(224, 224)) -> tuple[np.ndarray, list[str]]:
    """
    Genera embeddings para todas las imágenes en un directorio, procesándolas por lotes.

    Parámetros:
        image_dir (str): Ruta al directorio que contiene las imágenes (.jpg, .jpeg, .png).
        model_name (str): Nombre del modelo CLIP a utilizar con SentenceTransformer. Por defecto: 'clip-ViT-B-32'.
        batch_size (int): Tamaño de cada lote de imágenes para procesar.
        resize_to (tuple): Dimensiones (ancho, alto) a las que se redimensionarán las imágenes.

    Retorna:
        tuple: 
            - embeddings (np.ndarray): Array de embeddings generados (shape: [n_imágenes, dim]).
            - image_paths (list[str]): Lista con las rutas de las imágenes procesadas.
    """
    
    # Crear el modelo y cargarlo a GPU
    model = SentenceTransformer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Usando dispositivo: {device}")

    # Leer rutas de imagen}
    valid_exts = ('.jpg', '.jpeg', '.png')
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]

    all_embeddings = []

    # Procesar por lotes
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


def generate_image_embedding(image_path: str, model_name='clip-ViT-B-32', resize_to=(224, 224)) -> np.ndarray:
    """
    Genera un embedding para una sola imagen.

    Parámetros:
        image_path (str): Ruta a la imagen.
        model_name (str): Modelo CLIP a utilizar.
        resize_to (tuple): Tamaño al que se redimensionará la imagen.

    Retorna:
        np.ndarray: Vector de embedding (shape: [dim]).
    """
    model = SentenceTransformer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    try:
        img = Image.open(image_path).convert('RGB').resize(resize_to)
    except Exception as e:
        raise ValueError(f"No se pudo abrir la imagen: {e}")

    embedding = model.encode([img], convert_to_numpy=True, device=device)[0]  # [0] para retornar el vector plano
    return embedding

def combine_embeddings(txt_embeddings, img_embeddings, alpha=0.5):
    combined_embeddings = []
    for text_vec, image_vec in zip(txt_embeddings, img_embeddings):
        text_vec = np.array(text_vec)
        image_vec = np.array(image_vec)

        # Normalizar
        text_vec /= np.linalg.norm(text_vec)
        image_vec /= np.linalg.norm(image_vec)

        combined = alpha * text_vec + (1 - alpha) * image_vec
        combined /= np.linalg.norm(combined)  # Normalizar el embedding combinado
        combined_embeddings.append(combined)
    
    return combined_embeddings
