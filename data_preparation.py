import os
import numpy as np
import librosa
import random
from sklearn.model_selection import train_test_split
import librosa.display
import matplotlib.pyplot as plt

# Папка с данными
DATA_DIR = 'data/archive/'
OUTPUT_DIR = 'processed_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Эмоции, которые нас интересуют
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Преобразование строковых меток в числовые
EMOTIONS_TO_NUM = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6,
    'surprised': 7
}

# Список всех файлов в датасете
files = []

# Собираем список всех файлов
for actor_folder in os.listdir(DATA_DIR):
    actor_path = os.path.join(DATA_DIR, actor_folder)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                files.append(os.path.join(actor_path, file))

# Перемешиваем файлы
random.shuffle(files)

# Подготовка данных (признаков и меток)
features = []
labels = []

for file in files:
    # Извлекаем характеристики из имени файла
    filename = os.path.basename(file)
    parts = filename.split('-')
    
    emotion = EMOTIONS[parts[2]]  # Строковое значение эмоции
    emotion_label = EMOTIONS_TO_NUM[emotion]  # Преобразование в числовое значение
    
    # Загружаем аудио файл
    y, sr = librosa.load(file, sr=None)
    
    # Извлекаем признаки, например, MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_scaled = np.mean(mfccs.T, axis=0)  # Преобразуем в 1D вектор
    
    # Добавляем признаки и метку в соответствующие списки
    features.append(mfccs_scaled)
    labels.append(emotion_label)

# Преобразуем в массивы numpy
features = np.array(features)
labels = np.array(labels)

# Разбиваем данные на тренировочные, валидационные и тестовые наборы
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Сохраняем данные в файлы .npy
np.save(os.path.join(OUTPUT_DIR, "train_features.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "train_labels.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "val_features.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "val_labels.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "test_features.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "test_labels.npy"), y_test)

# Выводим информацию о количестве данных
print("Подготовка данных...")
print(f"Собрано {len(files)} файлов.")
print(f"Тренировочный: {len(X_train)}, Валидационный: {len(X_val)}, Тестовый: {len(X_test)}")
print(f"Данные сохранены в папке {OUTPUT_DIR}")
