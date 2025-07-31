import json
import pandas as pd

# Carga los captions desde un archivo JSON del dataset
def load_captions(json_path: str) -> pd.DataFrame:

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Construye un diccionario para mapear ID de imagen a su nombre de archivo
    image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}

    # Extrae pares (file_name, caption) de las anotaciones
    rows = []
    for ann in data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        filename = image_id_to_filename.get(image_id)

        # Solo añade si existe el nombre de archivo para la imagen
        if filename:
            rows.append((filename, caption))

    # Devuelve un DataFrame con los datos recopilados
    df = pd.DataFrame(rows, columns=['file_name', 'caption'])
    return df

# Obtiene el caption asociado a una imagen a partir del nombre del archivo
def get_caption_by_image_name(image_name: str, df: pd.DataFrame) -> str:
    try:
        # Busca la fila cuyo file_name coincida con image_name y devuelve el caption
        return df.loc[df['file_name'] == image_name]['caption'].values[0]
    except (KeyError, IndexError):
        return "Descripción no encontrada."
