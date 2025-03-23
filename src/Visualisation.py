import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import scrolledtext
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input

# Constants for segmentation model
SEG_MODEL_PATH = r'segmentation_95_05.h5'
IMG_HEIGHT = 256 
IMG_WIDTH = 256

# Constants for classification model
CLASS_MODEL_PATH = r'classifier.h5'
IMG_SIZE = 224
LABELS = ['benign', 'malignant', 'normal']

# Load the segmentation model
if os.path.exists(SEG_MODEL_PATH):
    print(f"Загрузка модели сегментации из {SEG_MODEL_PATH}")
    seg_model = load_model(SEG_MODEL_PATH)
else:
    raise FileNotFoundError(f"Модель сегментации не найдена по пути: {SEG_MODEL_PATH}")

# Load the classification model
if os.path.exists(CLASS_MODEL_PATH):
    print(f"Загрузка модели классификации из {CLASS_MODEL_PATH}")
    class_model = tf.keras.models.load_model(CLASS_MODEL_PATH)
else:
    raise FileNotFoundError(f"Модель классификации не найдена по пути: {CLASS_MODEL_PATH}")

def preprocess_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = img_to_array(image) / 255.0
    return image

def predict_mask(image):
    image = np.expand_dims(image, axis=0)
    prediction = seg_model.predict(image)
    processed_mask = (prediction[0] > 0.5).astype(np.uint8) * 255
    return processed_mask

def draw_contours(original_image, predicted_mask):
    contours, _ = cv2.findContours(predicted_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_image = cv2.cvtColor(original_image.squeeze(), cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        cv2.drawContours(overlay_image, [contour], -1, (0, 0, 255), 2) 

    return overlay_image

def combine_image_mask(original_image, predicted_mask):
    overlay_image = draw_contours(original_image, predicted_mask)
    combined_image = cv2.addWeighted(overlay_image, 0.7, cv2.cvtColor(original_image.squeeze(), cv2.COLOR_GRAY2BGR), 0.3, 0)
    return combined_image

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

def load_original_mask(img_path):
    mask_path = img_path.replace(".png", "_mask.png").replace(".jpg", "_mask.png").replace(".jpeg", "_mask.png")
    original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if original_mask is not None:
        original_mask = cv2.resize(original_mask, (IMG_WIDTH, IMG_HEIGHT))
        return original_mask
    else:
        raise FileNotFoundError(f"Оригинальная маска не найдена по пути: {mask_path}")

def visualize(image, predicted_mask, original_mask, predicted_class):
    overlay_image = draw_contours(image, predicted_mask)

    plt.figure(figsize=(24, 6))

    plt.subplot(1, 4, 1)
    plt.title("Оригинальное изображение")
    plt.imshow(image.squeeze(), cmap='gray')

    plt.subplot(1, 4, 2)
    plt.title("Оригинальная маска")
    plt.imshow(original_mask.squeeze(), cmap='gray')

    plt.subplot(1, 4, 3)
    plt.title("Предсказанная маска")
    plt.imshow(predicted_mask.squeeze(), cmap='gray')

    plt.subplot(1, 4, 4)
    plt.title("Границы предсказанного новообразования")
    plt.imshow(overlay_image)
    plt.axis('off')
    
    plt.suptitle(f'Предсказанный класс: {predicted_class[0]}, Доверие: {predicted_class[1]:.2f}')
    plt.show()

def select_images():
    file_paths = filedialog.askopenfilenames(title="Выберите изображения", filetypes=[("Файлы изображений", "*.png;*.jpg;*.jpeg")])
    if file_paths:
        output_text.delete(1.0, tk.END)
        for img_path in file_paths:
            try:
                # Сегментация
                image = preprocess_image(img_path)
                predicted_mask = predict_mask(image)
                combined_image = combine_image_mask(image, predicted_mask)

                # Загрузка оригинальной маски
                original_mask = load_original_mask(img_path)

                # Классификация
                img_array = load_and_preprocess_image(img_path)
                predicted_class = predict_class(img_array)

                # Визуализация
                visualize(image, predicted_mask, original_mask, predicted_class)
                
                # Вывод информации о классификации
                output_text.insert(tk.END, f'Изображение: {os.path.basename(img_path)} - Предсказанный класс: {predicted_class[0]}, Доверие: {predicted_class[1]:.2f}\n')
            except Exception as e:
                output_text.insert(tk.END, f'Ошибка при обработке {os.path.basename(img_path)}: {str(e)}\n')

# Create the main application window
app = tk.Tk()
app.title("Сегментация и классификация изображений")
app.geometry("600x400")

select_button = tk.Button(app, text="Выбрать изображения", command=select_images)
select_button.pack(pady=20)

output_text = scrolledtext.ScrolledText(app, width=70, height=15)
output_text.pack(pady=10)

app.mainloop()
