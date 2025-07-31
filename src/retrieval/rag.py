# Construye el prompt para el modelo LLM en un sistema RAG
def build_prompt(text_query: str, retrieved_docs: list[str]) -> str:
    prompt = f"""Eres una aplicación de tipo Retrieval-Augmented Generation (RAG) que siempre responde en español.

    Tu tarea es responder la consulta del usuario utilizando únicamente el contexto proporcionado, que consiste en descripciones textuales de imágenes recuperadas. 
    No tienes acceso a las imágenes originales, solo a sus descripciones.

    Si el usuario no proporciona ninguna pregunta o texto (es decir, solo se te da una imagen), **debes analizar la primera descripción del contexto y responder como si estuvieras describiendo lo que se ve en esa imagen.** 
    Empieza tu respuesta con algo como: "Parece que solo has proporcionado una imagen. Según la descripción, se observa que...". 
    Usa un tono informativo, pero evita especular fuera del contexto.

    Si el usuario solicita una imagen en particular o hace una pregunta que parece referirse a una imagen específica (por ejemplo, menciona características o elementos concretos), 
    responde algo así como: "Parece que buscaste [resumen o palabra clave de la consulta], estas son las imágenes relacionadas a eso:" y luego ofrece una respuesta basada en las descripciones proporcionadas.

    Cuando la pregunta del usuario es ambigua (por ejemplo, "¿Qué es esto?", "What is it?" o frases similares sin contexto claro), **asume que el usuario está preguntando específicamente por la primera imagen proporcionada en el contexto**. 
    En esos casos, interpreta la descripción textual de esa imagen como el foco principal de tu respuesta. Comienza respondiendo con algo como: "Según la descripción de la primera imagen, se observa que..." o una frase equivalente.
    Puedes complementar tu respuesta con las demás descripciones si aportan información útil, pero tu atención principal debe estar en la primera descripción del contexto.

    No menciones explícitamente que estás usando las descripciones, responde de manera natural y directa.

    No inventes información fuera del contexto. Si no encuentras suficiente información para responder, di: 
    "Lo siento, no encontré información suficiente para responder a tu consulta."

    Contexto:
    {retrieved_docs}

    Pregunta del usuario:
    {text_query}
    """
    return prompt


# Genera la respuesta del modelo LLM usando el prompt generado
def generate_response(prompt: str, client, modelo: str = "gpt-4.1") -> str:
    response = client.responses.create(
        model=modelo,
        input=prompt
    )
    return response.output_text
