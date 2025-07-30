import json
import pandas as pd
# import os
# from PIL import Image
# from torchvision import transforms

def load_captions(json_path: str) -> pd.DataFrame:
    """
    Carga descripciones (captions) del dataset desde un archivo JSON
    y las devuelve en un DataFrame de pandas.

    Args:
        json_path (str): Ruta al archivo captions_*.json.

    Returns:
        pd.DataFrame: DataFrame con columnas 'file_name' y 'caption'.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Mapeo de ID de imagen a nombre de archivo
    image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}

    # Recolección de datos para el DataFrame
    rows = []
    for ann in data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        filename = image_id_to_filename.get(image_id)

        if filename:
            rows.append((filename, caption))

    df = pd.DataFrame(rows, columns=['file_name', 'caption'])
    return df


def get_caption_by_image_name(image_name: str, df: pd.DataFrame) -> str:
    """
    Retorna el caption (descripción) asociado a una imagen a partir de su nombre.
    
    Parámetros:
    - image_name: nombre del archivo o ID de imagen (ej. "000000123456.jpg")
    - df: DataFrame con columnas ['image_id', 'caption']
    
    Retorna:
    - caption como string si se encuentra, si no, un mensaje de error.
    """
    try:
        return df.loc[image_name]["caption"]
    except KeyError:
        return "Descripción no encontrada."
