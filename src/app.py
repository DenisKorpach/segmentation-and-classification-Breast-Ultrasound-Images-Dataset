import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Пути к моделям
SEG_MODEL_PATH = r'segmentation_95_05.h5'
CLASS_MODEL_PATH = r'working\classifier.h5'
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_SIZE = 224
LABELS = ['доброкачественное', 'злокачественное', 'норма']

# Загрузка моделей
seg_model = load_model(SEG_MODEL_PATH)
class_model = load_model(CLASS_MODEL_PATH)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img_to_array(img) / 255.0
    return img

def predict_mask(image):
    image = np.expand_dims(image, axis=0)
    prediction = seg_model.predict(image)
    return (prediction[0] > 0.5).astype(np.uint8) * 255

def draw_contours(original_image, predicted_mask):
    contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_image = cv2.cvtColor(original_image.squeeze(), cv2.COLOR_GRAY2BGR)
    for contour in contours:
        cv2.drawContours(overlay_image, [contour], -1, (0, 0, 255), 2)
    return overlay_image

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_class(img_array):
    predictions = class_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return LABELS[predicted_class_index], predictions[0][predicted_class_index]

def generate_report(predicted_class, confidence):
    report = f"Отчет о результате анализа\n"
    report += f"Дата и время: {datetime.now()}\n"
    report += f"Предсказанный класс: {predicted_class}\n"
    report += f"Уверенность: {confidence:.2f}\n"
    report += "\nРекомендации:\n"
    report += "1. Результаты могут быть неточными. Рекомендуется обратиться к специалисту.\n"
    report += "2. Следите за состоянием и регулярно проходите медицинские осмотры.\n"
    report += "3. Если есть какие-либо симптомы, обратитесь к врачу.\n"
    return report

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("Файл не найден")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("Файл не выбран")
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(img_path)

            # Сегментация и классификация
            try:
                image = preprocess_image(img_path)
                predicted_mask = predict_mask(image)
                overlay_image = draw_contours(image, predicted_mask)
                img_array = load_and_preprocess_image(img_path)
                predicted_class, confidence = predict_class(img_array)

                # Генерация отчета
                report = generate_report(predicted_class, confidence)
                report_filename = f"report_{filename.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

                # Визуализация
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].imshow(image.squeeze(), cmap='gray')
                axs[0].set_title("Оригинальное изображение")
                axs[1].imshow(overlay_image)
                axs[1].set_title(f'Класс: {predicted_class}, Уверенность: {confidence:.2f}')
                
                # Конвертация изображения в base64
                img_io = BytesIO()
                plt.savefig(img_io, format="png")
                img_io.seek(0)
                img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

                return render_template("result.html", img_data=img_base64, predicted_class=predicted_class, confidence=confidence, report=report, report_filename=report_filename)
            except Exception as e:
                flash(f"Ошибка при обработке изображения: {str(e)}")
                return redirect(request.url)
    return render_template("upload.html")

@app.route("/download_report/<filename>", methods=["GET"])
def download_report(filename):
    report_content = request.args.get('report')
    response = app.response_class(
        response=report_content,
        status=200,
        mimetype='text/plain',
    )
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response

# Запуск приложения
if __name__ == "__main__":
    app.run(debug=True)
