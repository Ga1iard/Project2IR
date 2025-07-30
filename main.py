# from src.ui.app import demo

# if __name__ == "__main__":
#     demo.launch()

import os
import pandas as pd
import numpy as np
from openai import OpenAI
from src.retrieval.dataset_loader import load_captions
from src.retrieval.preprocessing import preprocess_documents, merge_captions_by_image
from src.retrieval.embedding_generation import generate_text_embeddings, generate_image_embeddings, combine_embeddings
from src.retrieval.vector_db import build_vector_index, save_index
from src.retrieval.rag import build_prompt
from src.retrieval.search__engine import retrieve_by_text, retrieve_by_image, retrieve_by_text_and_image

# Rutas de carga
IMG_PATH = 'src/data/test'
TXT_PATH = 'src/data/annotations/captions_val2014.json'

# Número de documentos relevantes a recuperar por consulta
TOP_K = 5

# 1. Cargar el corpus original (varios captions por imagen)
df_raw = load_captions(TXT_PATH).head(104)

# 2. Crear df combinando captions por imagen
df = merge_captions_by_image(df_raw, image_col='file_name', caption_col='caption')

# 3. Preprocesamiento
df_processed = preprocess_documents(df['combined_caption'].tolist())
df['prep_doc'] = df_processed['prep_doc']
del df_processed

# 4. Generar embeddings de texto
txt_embeddings = generate_text_embeddings(df['prep_doc'].tolist())
df['txt_embedding'] = [vec for vec in txt_embeddings]

# 5. Generar embeddings de imagen
img_embeddings, image_paths = generate_image_embeddings(IMG_PATH, batch_size=128)
image_names = [os.path.basename(path) for path in image_paths]
df_img = pd.DataFrame({
    'file_name': image_names,
    'img_embedding': list(img_embeddings)
})

# 6. Unir los embeddings de imagen
df = df.merge(df_img, on='file_name', how='left')
del df_img

# 7. Mostrar resultado
print(df.head())

# --- COMBINACIÓN DE EMBEDDINGS (MULTIMODAL) ---
combined_embeddings = combine_embeddings(df['txt_embedding'], df['img_embedding'], alpha=0.5)

# --- ÍNDICE FAISS MULTIMODAL ---
combined_embeddings_np = np.array(combined_embeddings).astype('float32')
index_mm = build_vector_index(combined_embeddings_np)
save_index(index_mm, path="index_multimodal.faiss")
print(index_mm)

# --- CONSULTA SOLO TEXTO ---
query_text = "a group of people riding horses"
_, topk_img_idxs = retrieve_by_text(query_text, index_mm, TOP_K)

print(f"\n[Consulta por texto] '{query_text}'")
for idx in topk_img_idxs[0]:
    file_name = df.iloc[idx]['file_name']
    caption = df.iloc[idx]['combined_caption']
    print(f" - Imagen: {file_name} | Caption: {caption}")

# --- CONSULTA SOLO IMAGEN ---
query_image_path = 'src/data/test/COCO_val2014_000000179765.jpg'
_, topk_img_idxs = retrieve_by_image(query_image_path, index_mm, TOP_K)

print(f"\n[Consulta por imagen] '{os.path.basename(query_image_path)}'")
for idx in topk_img_idxs[0]:
    file_name = df.iloc[idx]['file_name']
    caption = df.iloc[idx]['combined_caption']
    print(f" - Imagen: {file_name} | Caption: {caption}")

# --- CONSULTA TEXTO + IMAGEN ---
query_text = "what is it?"
query_image_path = 'src/data/test/COCO_val2014_000000179765.jpg'
_, topk_img_idxs = retrieve_by_text_and_image(query_text, query_image_path, index_mm, TOP_K)

print(f"\n[Consulta combinada] Texto: '{query_text}' | Imagen: '{os.path.basename(query_image_path)}'")
for idx in topk_img_idxs[0]:
    file_name = df.iloc[idx]['file_name']
    caption = df.iloc[idx]['combined_caption']
    print(f" - Imagen: {file_name} | Caption: {caption}")

# if __name__ == "__main__":
#     main()













#######
# Script de copia de imágenes
#####


# import os
# import shutil
# import pandas as pd

# # 1. Ruta a la carpeta con todas las imágenes (de COCO o similar)
# images_dir = 'src/data/val2014'  # Cambia esta ruta si es necesario

# # 2. Carpeta destino donde copiarás las imágenes
# test_dir = 'src/data/test'
# os.makedirs(test_dir, exist_ok=True)  # Crear si no existe

# # 3. Suponiendo que ya tienes el DataFrame cargado como 'df'
# # Asegúrate de que 'file_name' esté en el DataFrame
# file_names_to_copy = df['file_name'].head(100).tolist()

# # 4. Copiar imágenes
# copied = 0
# for file_name in file_names_to_copy:
#     src_path = os.path.join(images_dir, file_name)
#     dst_path = os.path.join(test_dir, file_name)

#     if os.path.exists(src_path):
#         shutil.copyfile(src_path, dst_path)
#         copied += 1
#     else:
#         print(f"[⚠️] No se encontró la imagen: {src_path}")

# print(f"[✅] Imágenes copiadas: {copied} de {len(file_names_to_copy)}")

