import os
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from src.retrieval.dataset_loader import load_captions
from src.retrieval.preprocessing import preprocess_documents, merge_captions_by_image
from src.retrieval.embedding_generation import generate_text_embeddings, generate_image_embeddings, combine_embeddings
from src.retrieval.vector_db import build_vector_index, save_index
from src.retrieval.rag import build_prompt
from src.retrieval.search__engine import retrieve_by_text, retrieve_by_image, retrieve_by_text_and_image

from src.ui.app import launch_ui

# Cargar API key y crear cliente OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Parámetros
IMG_PATH = 'src/data/test'
TXT_PATH = 'src/data/annotations/captions_val2014.json'
TOP_K = 5

# 1. Cargar y preparar datos
df_raw = load_captions(TXT_PATH).head(104)
df = merge_captions_by_image(df_raw, image_col='file_name', caption_col='caption')
df_processed = preprocess_documents(df['combined_caption'].tolist())
df['prep_doc'] = df_processed['prep_doc']

txt_embeddings = generate_text_embeddings(df['prep_doc'].tolist())
df['txt_embedding'] = [vec for vec in txt_embeddings]

img_embeddings, image_paths = generate_image_embeddings(IMG_PATH, batch_size=128)
image_names = [os.path.basename(path) for path in image_paths]
df_img = pd.DataFrame({'file_name': image_names, 'img_embedding': list(img_embeddings)})
df = df.merge(df_img, on='file_name', how='left')

combined_embeddings = combine_embeddings(df['txt_embedding'], df['img_embedding'], alpha=0.5)
index_mm = build_vector_index(np.array(combined_embeddings).astype('float32'))
save_index(index_mm, path="index_multimodal.faiss")

# 2. Lanzar UI pasando el cliente OpenAI
launch_ui(
    df,
    index_mm,
    IMG_PATH,
    TOP_K,
    build_prompt,
    retrieve_by_text,
    retrieve_by_image,
    retrieve_by_text_and_image,
    client  # <-- pasa el cliente aquí
)
