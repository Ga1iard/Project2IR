import gradio as gr
import os
from PIL import Image

def launch_ui(df, index_mm, IMG_PATH, TOP_K, build_prompt, retrieve_by_text, retrieve_by_image, retrieve_by_text_and_image, client):

    os.makedirs("dummy_images", exist_ok=True)  # carpeta temporal para imágenes

    def save_temp_image(image_np):
        img = Image.fromarray(image_np.astype('uint8'), 'RGB')
        save_path = os.path.join("dummy_images", "temp_query_image.png")
        img.save(save_path)
        return save_path

    def generate_response(prompt: str):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres una aplicación de tipo Retrieval-Augmented Generation (RAG) que siempre responde en español."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.2,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Error llamando al LLM: {e}"

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

        resultados = []
        for idx in topk_idxs[0]:
            row = df.iloc[idx]
            resultados.append((os.path.join(IMG_PATH, row['file_name']), row['combined_caption']))

        retrieved_docs = [caption for _, caption in resultados]

        prompt = build_prompt(query_text or "", retrieved_docs)
        response = generate_response(prompt)

        selected_img_path, selected_caption = resultados[0]
        return response, resultados, selected_img_path, selected_caption

    def on_click_image(image_path):
        file_name = os.path.basename(image_path)
        caption = df[df['file_name'] == file_name]['combined_caption'].values[0]
        return image_path, caption

    with gr.Blocks(css="""
        html, body, .gradio-container, .gr-blocks {
            height: 100vh;
            margin: 0;
        }
        body {
            background-color: #121212;
            color: #FFFFFF;
            font-family: 'Segoe UI', sans-serif;
        }
        .search-bar {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            background: #1e1e1e;
            border-radius: 15px;
            padding: 10px;
            height: 100%;
            gap: 10px;
        }
        .response-box-container {
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .response-box textarea {
            flex: 1;
            resize: none !important;
        }
        .image-preview {
            flex-grow: 1;
            min-height: 150px;
        }
        .image-preview img {
            height: 100% !important;
            width: 100% !important;
            object-fit: contain;
            border-radius: 8px;
        }
    """) as demo:
        query_text = gr.State("")
        uploaded_image = gr.State(None)

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

            with gr.Column(scale=2, elem_classes=["response-box-container"]):
                response_box = gr.Textbox(
                    label="Respuesta del LLM",
                    interactive=False,
                    lines=15,
                    elem_classes=["response-box"]
                )

        with gr.Row():
            with gr.Column(scale=12):
                gallery = gr.Gallery(label="Top-K Imágenes", columns=5, rows=1, height="auto", interactive=False)

        def submit(query, image):
            response, topk, selected_path, caption = search_rag(query, image)
            return response, topk

        search_btn.click(
            fn=submit,
            inputs=[input_text, img_upload],
            outputs=[response_box, gallery]
        )

        gallery.select(fn=on_click_image, inputs=[gallery], outputs=[])

    demo.launch()
