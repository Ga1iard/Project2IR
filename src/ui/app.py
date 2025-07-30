import os
import gradio as gr
from PIL import Image

def launch_ui(df, index_mm, IMG_PATH, TOP_K, build_prompt, retrieve_by_text, retrieve_by_image, retrieve_by_text_and_image):
    """
    Lanza la interfaz Gradio para el sistema RAG multimodal.
    """
    def search_rag(query_text, query_image):
        if query_text and query_image:
            _, topk_idxs = retrieve_by_text_and_image(query_text, query_image, index_mm, TOP_K)
        elif query_text:
            _, topk_idxs = retrieve_by_text(query_text, index_mm, TOP_K)
        elif query_image:
            _, topk_idxs = retrieve_by_image(query_image, index_mm, TOP_K)
        else:
            return "Escribe algo o sube una imagen", [], None, None

        resultados = []
        for idx in topk_idxs[0]:
            row = df.iloc[idx]
            resultados.append((os.path.join(IMG_PATH, row['file_name']), row['combined_caption']))

        selected_img_path, selected_caption = resultados[0]
        response = build_prompt(query_text, selected_caption)
        return response, resultados, selected_img_path, selected_caption

    def on_click_image(image_path):
        file_name = os.path.basename(image_path)
        caption = df[df['file_name'] == file_name]['combined_caption'].values[0]
        return image_path, caption

    with gr.Blocks(theme=gr.themes.Soft(), css="body { background-color: #121212; color: white; }") as demo:
        with gr.Row():
            with gr.Column():
                query_text = gr.Textbox(label="Buscar", placeholder="Escribe tu pregunta...", scale=8)
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="", show_label=False, interactive=True)
            with gr.Column(scale=1):
                search_btn = gr.Button("🔍", elem_id="search_btn")

        preview = gr.Image(label="Previsualización", visible=False)
        response_box = gr.Textbox(label="Respuesta del LLM", lines=3, interactive=False)

        with gr.Row():
            with gr.Column(scale=3):
                gallery = gr.Gallery(label="Top-K Imágenes", columns=5, rows=1, height="auto")
            with gr.Column(scale=2):
                selected_img = gr.Image(label="Imagen Seleccionada")
                selected_caption = gr.Textbox(label="Caption", lines=2, interactive=False)

        def on_submit(query, image):
            response, topk, selected_path, caption = search_rag(query, image)
            preview.visible = False
            return response, topk, selected_path, caption

        search_btn.click(
            fn=on_submit,
            inputs=[query_text, image_input],
            outputs=[response_box, gallery, selected_img, selected_caption]
        )

        gallery.select(fn=on_click_image, inputs=[gallery], outputs=[selected_img, selected_caption])

    demo.launch()
