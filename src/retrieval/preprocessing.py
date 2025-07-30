import nltk
import pandas as pd
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descarga de recursos necesarios de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def remove_stopwords(tokens):
    """
    Elimina las palabras vacías (stopwords) de una lista de tokens.

    Parámetros:
        tokens (list[str]): Lista de tokens.

    Retorna:
        list[str]: Lista de tokens sin stopwords.
    """
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def lemmatize_tokens(tokens):
    """
    Lematiza una lista de tokens utilizando WordNetLemmatizer.

    Parámetros:
        tokens (list[str]): Lista de tokens.

    Retorna:
        list[str]: Lista de lemas correspondientes.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_documents(documents, return_type='df'):
    """
    Preprocesa una lista de documentos aplicando minúsculas, tokenización, eliminación de stopwords y lematización.

    Parámetros:
        documents (list[str]): Lista de textos crudos.
        return_type (str): 'df' para retornar un DataFrame, 'tokens' para retornar solo las listas de lemas.

    Retorna:
        pd.DataFrame | list[list[str]]: DataFrame con los documentos originales y preprocesados, 
                                        o lista de listas de tokens si se especifica 'tokens'.
    """
    # Crear DataFrame con los documentos originales
    df = pd.DataFrame(documents, columns=['document'])
    
    # Tokenización con expresión regular y conversión a minúsculas
    df['regex_tokens'] = df['document'].str.lower().apply(
        lambda text: regexp_tokenize(text, pattern=r'\w[a-z]+')
    )

    # Eliminación de stopwords
    df['no_stopwords'] = df['regex_tokens'].apply(remove_stopwords)

    # Lematización
    df['lemmas'] = df['no_stopwords'].apply(lemmatize_tokens)

    # Unir los lemas en una cadena de texto
    df['prep_doc'] = df['lemmas'].str.join(' ')

    # Retornar según el tipo solicitado
    if return_type == 'tokens':
        return df['lemmas'].tolist()
    else:
        return df[['document', 'prep_doc']]


def preprocess_both(text: str) -> tuple[str, list[str]]:
    """
    Preprocesa un solo documento y retorna tanto el texto limpio como los tokens.

    Parámetros:
        text (str): Texto crudo.

    Retorna:
        tuple:
            - str: Documento preprocesado como texto limpio.
            - list[str]: Tokens lematizados del documento.
    """
    df = preprocess_documents([text])
    clean = df['prep_doc'].iloc[0]
    tokens = preprocess_documents([text], return_type='tokens')[0]
    return clean, tokens


def merge_captions_by_image(df, image_col='image_path', caption_col='caption'):
    """
    Agrupa los captions por imagen, combinando todos los textos en uno solo por imagen.

    Parámetros:
        df (pd.DataFrame): DataFrame con al menos una columna de imagen y una de caption.
        image_col (str): Nombre de la columna que contiene la ruta o ID de la imagen.
        caption_col (str): Nombre de la columna que contiene los captions.

    Retorna:
        pd.DataFrame: Nuevo DataFrame con una fila por imagen y captions combinados.
    """
    merged_df = df.groupby(image_col)[caption_col].apply(lambda captions: ' '.join(captions)).reset_index()
    merged_df.rename(columns={caption_col: 'combined_caption'}, inplace=True)
    return merged_df
