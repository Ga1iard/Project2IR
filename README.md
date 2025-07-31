# Sistema Retrieval Augmented Generation Multimodal

Este proyecto implementa un sistema de recuperación aumentada por recuperación (RAG) que permite ingresar texto, imágenes o ambos, y genera respuestas usando un modelo LLM (GPT-4) a partir de los documentos e imágenes más relevantes.

Se utilizan embeddings de texto e imagen para construir un índice FAISS multimodal, y una interfaz gráfica en Gradio para la interacción con el usuario.

## Estructura del Proyecto

```
src/
├── retrieval/
│   ├── dataset_loader.py       # Carga captions e IDs desde archivo CSV
│   ├── preprocessing.py        # Preprocesamiento de captions e imágenes
│   ├── embedding_generation.py # Generación de embeddings para texto e imagen
│   ├── vector_db.py            # Indexación y búsqueda con FAISS
│   ├── rag.py                  # Construcción de prompts y respuestas con LLM
│   └── search_engine.py        # Recuperación por texto, imagen o ambos
├── ui/
│   └── app.py                  # Interfaz gráfica en Gradio

main.py                         # Script principal de ejecución
requirements.txt                # Lista de dependencias
README.md                       # Documentación del proyecto
```

## Instalación

1. **Clona el repositorio:**
    ```bash
   git clone https://github.com/Ga1iard/Project2IR.git
   cd Project2IR

2. **Descargar el dataset de COCO**
    https://cocodataset.org/#download

3. **Instala las dependencias:**
   pip install -r requirements.txt

4. **Asegurarse de tener un API key de OpenAI**
   Ingresar el mismo en un archivo .env

5. **Asegurarse de tener soporte GPU para FAISS:**
   pip install faiss-gpu

## Descarga de recursos necesarios de NLTK
Los mismos se descargan automáticamente, pero se los puede obtener mediante
  nltk.download('punkt', quiet=True)
  nltk.download('stopwords', quiet=True)
  nltk.download('wordnet', quiet=True)

## Uso
El sistema se ejecuta desde el archivo **main.py** o con **python main.py**

La aplicación permite ingresar texto, subir una imagen o ambos, y devuelve:
- Documentos e imágenes más relevantes.
- La imagen seleccionada con su descripción.
- Una respuesta generada por el LLM basada en la entrada y los documentos recuperados.

## Tecnologías
- CLIP (OpenAI) para embeddings multimodales
- FAISS (Facebook) para búsqueda vectorial
- LLM  (openAI) para RAG
- Gradio para la interfaz de usuario
- Pandas, NumPy, PIL, scikit-learn

## Notas finales
El dataset específico utilizado para la ejecución del presente proyecto es **2014 Val Images** y **2014 Train/Val anotations** de COCO
El proyecto **requiere de una GPU compatible con CUDA** para su ejecución.

## Integrantes (Grupo 4)
- Andrés Carrillo
- Daniel Lorences
- César Villacís
