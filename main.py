import os
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from src.retrieval.dataset_loader import load_captions
from src.retrieval.preprocessing import preprocess_documents, merge_captions_by_image
from src.retrieval.embedding_generation import generate_text_embeddings, generate_image_embeddings, combine_embeddings
from src.retrieval.vector_db import build_vector_index, save_index
from src.retrieval.rag import build_prompt, generate_response
from src.retrieval.search__engine import retrieve_by_text, retrieve_by_image, retrieve_by_text_and_image
from src.ui.app import launch_ui

# Configuración y cliente OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Definición de rutas y parámetros principales
IMG_PATH = 'src/data/val2014'
TXT_PATH = 'src/data/annotations/captions_val2014.json'
TOP_K = 10

# Carga de datos
df_raw = load_captions(TXT_PATH)
df = merge_captions_by_image(df_raw, image_col='file_name', caption_col='caption')

# Preprocesamiento de textos y generación de embeddings de texto
df_processed = preprocess_documents(df['combined_caption'].tolist())
df['prep_doc'] = df_processed['prep_doc']
txt_embeddings = generate_text_embeddings(df['prep_doc'].tolist())
df['txt_embedding'] = [vec for vec in txt_embeddings]

# Generación de embeddings de imagen y unión con DataFrame principal
img_embeddings, image_paths = generate_image_embeddings(IMG_PATH, batch_size=10024)
image_names = [os.path.basename(path) for path in image_paths]
df_img = pd.DataFrame({'file_name': image_names, 'img_embedding': list(img_embeddings)})
df = df.merge(df_img, on='file_name', how='left')

# Combinación de embeddings textuales y visuales para indexado multimodal
combined_embeddings = combine_embeddings(df['txt_embedding'], df['img_embedding'], alpha=0.5)
index_mm = build_vector_index(np.array(combined_embeddings).astype('float32'))
save_index(index_mm, path="index_multimodal.faiss")

# Lanzamiento de la interfaz gráfica con el sistema de recuperación multimodal
launch_ui(
    df,
    index_mm,
    IMG_PATH,
    TOP_K,
    build_prompt,
    retrieve_by_text,
    retrieve_by_image,
    retrieve_by_text_and_image,
    client,
    generate_response 
)