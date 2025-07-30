import gradio as gr
from PIL import Image

# Simulación de búsqueda (reemplaza con tu lógica de RAG multimodal)
def search(query, image):
    # Dummy outputs (reemplazar con resultado real)
    results = [
        "assets/cat1.jpg",
        "assets/cat2.jpg",
        "assets/cat3.jpg",
        "assets/cat4.jpg"
    ]
    description = "This is a cute cat with light brown and cream fur."
    return results, description

with gr.Blocks() as demo:
    with gr.Row():
        text_input = gr.Textbox(placeholder="Text Here...", label="", show_label=False)
        image_input = gr.Image(type="filepath", label="", elem_id="image-uploader")
        search_btn = gr.Button("🔍")

    gallery = gr.Gallery(label="Results")
    description_output = gr.Textbox(label="Description", lines=4)

    search_btn.click(fn=search, inputs=[text_input, image_input], outputs=[gallery, description_output])

