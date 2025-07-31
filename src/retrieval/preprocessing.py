import nltk
import pandas as pd
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descarga silenciosa de los recursos necesarios para tokenización, lematización y stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Función para eliminar palabras vacías (stopwords) de una lista de tokens
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

# Función para lematizar tokens con WordNetLemmatizer
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Preprocesamiento completo de una lista de documentos: limpieza, tokenización, stopwords y lematización
def preprocess_documents(documents, return_type='df'):
    # Crear DataFrame con documentos originales
    df = pd.DataFrame(documents, columns=['document'])
    
    # Tokenización con expresión regular y conversión a minúsculas
    df['regex_tokens'] = df['document'].str.lower().apply(
        lambda text: regexp_tokenize(text, pattern=r'\w[a-z]+')
    )

    # Eliminación de stopwords
    df['no_stopwords'] = df['regex_tokens'].apply(remove_stopwords)

    # Lematización de los tokens resultantes
    df['lemmas'] = df['no_stopwords'].apply(lemmatize_tokens)

    # Unir los lemas en un solo string (documento limpio)
    df['prep_doc'] = df['lemmas'].str.join(' ')

    # Devolver ya sea el DataFrame completo o solo los tokens
    if return_type == 'tokens':
        return df['lemmas'].tolist()
    else:
        return df[['document', 'prep_doc']]

# Preprocesamiento de un solo texto con salida doble: texto limpio y tokens lematizados
def preprocess_both(text: str) -> tuple[str, list[str]]:
    df = preprocess_documents([text])
    clean = df['prep_doc'].iloc[0]
    tokens = preprocess_documents([text], return_type='tokens')[0]
    return clean, tokens

# Agrupación de captions por imagen: concatena todos los captions asociados a la misma imagen
def merge_captions_by_image(df, image_col='image_path', caption_col='caption'):
    merged_df = df.groupby(image_col)[caption_col].apply(lambda captions: ' '.join(captions)).reset_index()
    merged_df.rename(columns={caption_col: 'combined_caption'}, inplace=True)
    return merged_df
