import gradio as gr
import os
from PIL import Image

# Función principal para lanzar la interfaz gráfica
def launch_ui(df, index_mm, IMG_PATH, TOP_K, build_prompt, retrieve_by_text, retrieve_by_image, retrieve_by_text_and_image, client, generate_response_fn):

    # Crear carpeta temporal para imágenes subidas
    os.makedirs("dummy_images", exist_ok=True)

    # Guardar la imagen cargada por el usuario como archivo temporal
    def save_temp_image(image_np):
        img = Image.fromarray(image_np.astype('uint8'), 'RGB')
        save_path = os.path.join("dummy_images", "temp_query_image.png")
        img.save(save_path)
        return save_path

    # Lógica principal de recuperación: por texto, por imagen o por ambos
    def search_rag(query_text, query_image):
        if query_text and query_image is not None:
            image_path = save_temp_image(query_image)
            _, topk_idxs = retrieve_by_text_and_image(query_text, image_path, index_mm, TOP_K)
        elif query_text:
            _, topk_idxs = retrieve_by_text(query_text, index_mm, TOP_K)
        elif query_image is not None:
            image_path = save_temp_image(query_image)
            _, topk_idxs = retrieve_by_image(image_path, index_mm, TOP_K)
        else:
            return "Escribe algo o sube una imagen", [], None, None

        # Preparar resultados a mostrar
        resultados = []
        for idx in topk_idxs[0]:
            row = df.iloc[idx]
            resultados.append((os.path.join(IMG_PATH, row['file_name']), row['combined_caption']))

        # Generar respuesta con el LLM a partir de los documentos recuperados
        retrieved_docs = [caption for _, caption in resultados]
        prompt = build_prompt(query_text or "", retrieved_docs)
        response = generate_response_fn(prompt, client)  # <<<< usar función externa

        selected_img_path, selected_caption = resultados[0]
        return response, resultados, selected_img_path, selected_caption

    # Manejador al seleccionar una imagen de la galería
    def on_click_image(selected):
        try:
            if isinstance(selected, list):
                image_path = selected[0]
            else:
                image_path = selected

            file_name = os.path.basename(image_path)
            caption = df[df['file_name'] == file_name]['combined_caption'].values[0]
            return image_path, caption
        except Exception as e:
            print(f"Error en el on click image: {e}")

    # Construcción de la interfaz con Gradio Blocks
    with gr.Blocks(css="""...""") as demo:
        query_text = gr.State("")
        uploaded_image = gr.State(None)

        # Zona de entrada de texto e imagen
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                with gr.Column(elem_classes=["search-bar"]):
                    input_text = gr.Textbox(show_label=False, placeholder="Busca algo...", lines=1)
                    img_upload = gr.Image(
                        label="",
                        interactive=True,
                        show_label=False,
                        visible=True,
                        elem_classes=["image-preview"]
                    )
                    search_btn = gr.Button("🔍")

            # Zona de respuesta del modelo
            with gr.Column(scale=2, elem_classes=["response-box-container"]):
                response_box = gr.Textbox(
                    label="Respuesta del LLM",
                    interactive=False,
                    lines=15,
                    elem_classes=["response-box"]
                )

        # Galería para mostrar resultados (top-k imágenes)
        with gr.Row():
            with gr.Column(scale=12):
                gallery = gr.Gallery(label="Top-K Imágenes", columns=5, rows=1, height="auto", interactive=False)

        # Función que ejecuta la búsqueda y genera la respuesta
        def submit(query, image):
            response, topk, selected_path, caption = search_rag(query, image)
            return response, topk

        # Asignar función al botón de búsqueda
        search_btn.click(
            fn=submit,
            inputs=[input_text, img_upload],
            outputs=[response_box, gallery]
        )

        # Asignar función al seleccionar imagen de la galería
        gallery.select(fn=on_click_image, inputs=[gallery], outputs=[])

    # Ejecutar la aplicación
    demo.launch()
