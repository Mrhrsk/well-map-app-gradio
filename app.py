import gradio as gr
import cv2
import numpy as np
import easyocr
from PIL import Image

# Инициализация EasyOCR с английским языком
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./models')

def process_image(image, target_name):
    # Преобразуем PIL-изображение в формат OpenCV
    image = np.array(image.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Используем исходное изображение без улучшения контрастности
    processed_image = image.copy()

    # Выполняем OCR на полном изображении
    results_full = reader.readtext(processed_image, allowlist='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    all_results = results_full.copy()

    # Метод скользящего окна (размер окна 600, шаг 300)
    window_size = 600
    step_size = 300
    height, width, _ = processed_image.shape

    for y in range(0, height - window_size + 1, step_size):
        for x in range(0, width - window_size + 1, step_size):
            section = processed_image[y:y + window_size, x:x + window_size]
            results = reader.readtext(section, allowlist='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
            for (bbox, text, prob) in results:
                global_bbox = [
                    (bbox[0][0] + x, bbox[0][1] + y),
                    (bbox[1][0] + x, bbox[1][1] + y),
                    (bbox[2][0] + x, bbox[2][1] + y),
                    (bbox[3][0] + x, bbox[3][1] + y)
                ]
                all_results.append((global_bbox, text, prob))

    # Аннотирование изображения
    found = False
    annotated_image = processed_image.copy()
    target_name_clean = target_name.strip().lower()

    for (bbox, text, prob) in all_results:
        detected_text = text.strip().lower()
        if target_name_clean in detected_text:
            found = True
            top_left = (int(bbox[0][0]), int(bbox[0][1]))
            bottom_right = (int(bbox[2][0]), int(bbox[2][1]))
            arrow_start = (top_left[0] - 70, top_left[1] - 120)
            arrow_end = (top_left[0], top_left[1])
            # Рисуем прямоугольник, стрелку и текст
            cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.arrowedLine(annotated_image, arrow_start, arrow_end, (0, 0, 255), 6, tipLength=0.35)
            cv2.putText(annotated_image, text, (top_left[0], top_left[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if not found:
        message = "Месторождение не найдено!"
    else:
        message = f"Месторождение '{target_name}' выделено на изображении."

    # Преобразуем изображение обратно в RGB для корректного отображения в Gradio
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return annotated_image, message

# CSS для установки полной ширины контейнера
css = """
.gradio-container {
    width: 100% !important;
}
"""

# Создаем интерфейс с использованием Gradio Blocks
with gr.Blocks(css=css, title="Карта месторождений") as demo:
    gr.Markdown("## Карта месторождений")
    with gr.Row():
        image_input = gr.Image(type="pil", label="Загрузите изображение карты (PNG или JPG)")
    text_input = gr.Textbox(label="Введите название месторождения для поиска", placeholder="например, hf1936")
    output_image = gr.Image(label="Обновлённое изображение карты")
    output_text = gr.Textbox(label="Результат")
    btn = gr.Button("Найти")
    btn.click(fn=process_image, inputs=[image_input, text_input], outputs=[output_image, output_text])

demo.launch(share=True)