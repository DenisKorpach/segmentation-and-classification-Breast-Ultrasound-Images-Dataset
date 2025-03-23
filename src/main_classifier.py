import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.optimizers import Adam

# начальное значение для воспроизводимости
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)

# Константы
DATA_DIR = r'Dataset_BUSI_with_GT'
OUTPUT_DIR = r'working'
IMG_SIZE = 224  # Размер изображений для входа в модель
BATCH_SIZE = 32  # Размер батча

# Метки классов
LABELS = ['benign', 'malignant', 'normal']

# Функция для наложения изображений и масок
def overlay_and_save(image_path, mask_path, output_path):
    if os.path.exists(image_path) and os.path.exists(mask_path):
        image = Image.open(image_path).convert('RGB')  # Конвертация в RGB
        mask = Image.open(mask_path).convert('RGB').resize(image.size)
        overlayed = Image.blend(image, mask, alpha=0.5)
        overlayed.save(os.path.join(output_path, os.path.basename(image_path)))

# Создаем необходимые директории для хранения данных
def create_output_dirs(output_dir):
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    for label in LABELS:
        os.makedirs(os.path.join(output_dir, 'train', label), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', label), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', label), exist_ok=True)

# Создаем выходные директории
create_output_dirs(OUTPUT_DIR)

# Накладываем изображения и маски в train, val и test
def process_images():
    all_data = []
    
    for label in LABELS:
        label_dir = os.path.join(DATA_DIR, label)
        for image_filename in os.listdir(label_dir):
            if image_filename.endswith('.png'):
                image_path = os.path.join(label_dir, image_filename)
                mask_path = os.path.join(label_dir, image_filename.replace('.png', '_mask.png'))
                all_data.append((image_path, mask_path, label))

    return all_data

# Получаем все данные
all_data = process_images()

# Разделяем данные на train, val и test
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=random_seed, stratify=[label for _, _, label in all_data])
train_data, val_data = train_test_split(train_data, test_size=0.125, random_state=random_seed, stratify=[label for _, _, label in train_data])

# Функция для сохранения наложенных изображений
def save_overlayed_images(data, output_subdir):
    for image_path, mask_path, label in data:
        overlay_and_save(image_path, mask_path, os.path.join(OUTPUT_DIR, output_subdir, label))

# Сохраняем наложенные изображения
save_overlayed_images(train_data, 'train')
save_overlayed_images(val_data, 'val')
save_overlayed_images(test_data, 'test')

# Создаем DataFrame для изображений и меток
def create_data_frame(output_dir):
    file_paths = []
    labels = []
    for subset in ['train', 'val', 'test']:
        for label in LABELS:
            label_dir = os.path.join(output_dir, subset, label)
            for image_file in os.listdir(label_dir):
                if image_file.endswith('.png'):
                    file_paths.append(os.path.join(label_dir, image_file))
                    labels.append(label)
    return pd.DataFrame({'Image_Path': file_paths, 'Label': labels})

# Загружаем данные
data = create_data_frame(OUTPUT_DIR)
train_data, test_data = train_test_split(data, test_size=0.1, random_state=random_seed, stratify=data['Label'])
train_data, val_data = train_test_split(train_data, test_size=0.085, random_state=random_seed, stratify=train_data['Label'])

# Создаем генераторы изображений
def create_generators():
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, 
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(
        train_data,
        x_col='Image_Path',
        y_col='Label',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=random_seed
    )

    validation_generator = val_datagen.flow_from_dataframe(
        val_data,
        x_col='Image_Path',
        y_col='Label',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=random_seed
    )

    return train_generator, validation_generator

train_generator, validation_generator = create_generators()

def build_model():
    base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    output = Dense(len(LABELS), activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output)

model = build_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Обучаем модель
history = model.fit(train_generator, validation_data=validation_generator, epochs=3)

# Сохраняем модель
model_save_path = os.path.join(OUTPUT_DIR, 'classifier_test.h5')
model.save(model_save_path)
print(f'Модель сохранена в {model_save_path}')

# Оценка модели
def evaluate_model():
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_dataframe(test_data,
                                                      x_col='Image_Path',
                                                      y_col='Label',
                                                      target_size=(IMG_SIZE, IMG_SIZE),
                                                      batch_size=BATCH_SIZE,
                                                      class_mode='categorical',
                                                      shuffle=False)
    
    test_loss, test_acc = model.evaluate(test_generator)
    print('Test accuracy:', test_acc)

    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    # Calculate F1 score and generate classification report
    f1 = f1_score(true_classes, predicted_classes, average='weighted')
    print('F1 Score:', f1)
    print(classification_report(true_classes, predicted_classes))

    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

evaluate_model()
