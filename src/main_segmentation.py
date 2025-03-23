import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import ReLU, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, Concatenate
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Пути к папкам с изображениями и масками
dataset_path = r"Dataset_BUSI_with_GT"
categories = ['benign', 'malignant', 'normal']

# Размеры изображений для обучения
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Функция для загрузки данных
def load_data():
    images = []
    masks = []
    
    for category in categories:
        path = os.path.join(dataset_path, category)
        for file in os.listdir(path):
            if '_mask' not in file:  # Загружаем изображения
                image = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                image = img_to_array(image) / 255.0
                images.append(image)
                
                # Загружаем маску
                mask_file = file.replace('.png', '_mask.png')
                mask = cv2.imread(os.path.join(path, mask_file), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
                mask = img_to_array(mask) / 255.0
                masks.append(mask)
    
    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    return images, masks

images, masks = load_data()

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.1, random_state=42)

print(f"Train images: {X_train.shape}, Train masks: {y_train.shape}")
print(f"Test images: {X_test.shape}, Test masks: {y_test.shape}")


model_path = 'segmentation_95_05h.h5'

if os.path.exists(model_path):
    # Загружаем сохранённую модель
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
else:
    class U_Net():
        def __init__(self, inp=(256, 256, 1), min_filter=64, max_filter=1024, optimizer='SGD', loss=BinaryCrossentropy(), metrics=['accuracy', AUC()]):
            self.max_filter = max_filter
            self.min_filter = min_filter
            self.img_inp = Input(shape=inp)
            
            down, down_list = self.downsample()
            
            lower, down_list = self.contracting_path(down, down_list)
 
            upper = self.upsample(lower, down_list)
            
            out = Conv2D(1, (1, 1), activation="sigmoid", name='output')(upper)
            self.out = out
            self.model = Model(inputs=self.img_inp, outputs=self.out)
            optimizer = SGD(learning_rate=0.01, momentum=0.9)
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        def downsample(self):
            k = self.min_filter
            down_list = []
            res = self.img_inp
            
            while k < self.max_filter:
                res = Conv2D(k, (3, 3), padding='same')(res)
                res = ReLU()(res)
                res = Conv2D(k, (3, 3), padding='same')(res)
                res = ReLU()(res)
                down_list.append(res)
                res = MaxPooling2D(pool_size=(2, 2), strides=2)(res)
                k *= 2
            
            return res, down_list

        def upsample(self, res, down_list):
            k = self.max_filter
            
            while k > self.min_filter:
                res = Conv2D(int(k/2), (3, 3), padding='same')(res)
                res = ReLU()(res)
                k /= 2
                
                if k > self.min_filter:
                    res = Conv2D(int(k/2), (3, 3), padding='same')(res)
                    res = ReLU()(res)
                    res = UpSampling2D((2, 2))(res)
                    left = down_list.pop()
                    res = self.crop_and_conc(left, res)
            
            return res

        def contracting_path(self, down, down_list):
            res = Conv2D(self.max_filter, (3, 3), padding='same')(down)
            res = ReLU()(res)
            res = Conv2D(int(self.max_filter/2), (3, 3), padding='same')(res)
            res = ReLU()(res)
            res = UpSampling2D((2, 2))(res)
            left = down_list.pop()
            res = self.crop_and_conc(left, res)
            return res, down_list
        
        def crop_and_conc(self, left, right):
            left = Cropping2D(cropping=int((left.shape[1] - right.shape[1]) / 2))(left)
            return Concatenate()([left, right])

    # Создаем модель и обучаем её
    shape = (256, 256, 1)
    model = U_Net(inp=shape).model
    model.summary()
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=80, batch_size=16)

    # Сохранение модели
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Пример предсказания
def postprocess_mask(prediction, threshold=0.5):
    processed_mask = (prediction > threshold).astype(np.uint8)
    return processed_mask * 255

def predict_image(image):
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_mask = postprocess_mask(prediction[0])
    return predicted_mask

def visualize(image, mask, prediction):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title(f"Original Image")
    plt.imshow(image.squeeze(), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title(f"True Mask")
    plt.imshow(mask.squeeze(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title(f"Predicted Mask")
    plt.imshow(prediction.squeeze(), cmap='gray')

    plt.show()

# Функция для обведения предсказанных сегментов в красную область
def draw_contours(original_image, predicted_mask):
    # Преобразуем предсказанную маску в бинарный формат
    contours, _ = cv2.findContours(predicted_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_image = cv2.cvtColor(original_image.squeeze(), cv2.COLOR_GRAY2BGR)  # Конвертируем в BGR для цветного изображения

    for contour in contours:
        cv2.drawContours(overlay_image, [contour], -1, (0, 0, 255), 2)  # Рисуем контуры красным цветом

    return overlay_image

# Тестирование модели на нескольких примерах
num_samples = 20  # Количество примеров для тестирования
for i in range(num_samples):
    image = X_test[(i+1)*3]
    mask = y_test[(i+1)*3]
    prediction = predict_image(image)
    visualize(image, mask, prediction)
    
    # Рисуем контуры на оригинальном изображении
    overlay_image = draw_contours(image, prediction)
    
    # Отображение изображения с контурами
    plt.figure(figsize=(6, 6))
    plt.title("Predicted Segments with Contours")
    plt.imshow(overlay_image)
    plt.axis('off')
    plt.show()
