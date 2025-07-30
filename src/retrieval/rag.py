
def build_prompt(text_query, retrieved_docs: list[str]) -> str:
        
    prompt = f"""Eres una aplicación de tipo Retrieval-Augmented Generation (RAG) que siempre responde en español.

    Tu tarea es responder la consulta del usuario utilizando únicamente el contexto proporcionado. 
    Este contexto puede incluir texto proveniente de descripciones de imágenes recuperadas. 
    No se proporciona la imagen real, solo su descripción textual.

    Debes basarte únicamente en el contenido del contexto para generar tu respuesta. 
    Si no encuentras suficiente información, responde con: "Lo siento, no encontré información suficiente para responder a tu consulta."

    Contexto:
    {retrieved_docs}

    Pregunta del usuario:
    {text_query}
    """

    return prompt



def generate_response(prompt: str, client, modelo: str = "gpt-4.1") -> str:
    response = client.responses.create(
        model=modelo,
        input=prompt
    )
    return response.output_text