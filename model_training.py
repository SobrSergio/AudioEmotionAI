# Определение классов эмоций
EMOTIONS = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust',
    7: 'surprised'
}

# Импортируем необходимые библиотеки
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import librosa

# Аугментация данных (пример: добавление шума, изменение скорости, изменение высоты тона)
def augment_data(features):
    augmented = []
    for feature in features:
        # Добавление шума
        noise = np.random.normal(0, 0.05, feature.shape)
        noisy_feature = feature + noise
        augmented.append(noisy_feature)

        # Изменение скорости
        try:
            stretched_feature = librosa.effects.time_stretch(feature, rate=1.1)
            if len(stretched_feature) > len(feature):
                stretched_feature = stretched_feature[:len(feature)]
            else:
                stretched_feature = np.pad(stretched_feature, (0, len(feature) - len(stretched_feature)), mode='constant')
            augmented.append(stretched_feature)
        except Exception as e:
            print(f"Ошибка при изменении скорости: {e}")

        # Изменение высоты тона
        try:
            pitched_feature = librosa.effects.pitch_shift(feature, sr=48000, n_steps=2)
            if len(pitched_feature) > len(feature):
                pitched_feature = pitched_feature[:len(feature)]
            else:
                pitched_feature = np.pad(pitched_feature, (0, len(feature) - len(pitched_feature)), mode='constant')
            augmented.append(pitched_feature)
        except Exception as e:
            print(f"Ошибка при изменении высоты тона: {e}")
    return np.array(augmented)

# Загрузка данных
X_train = np.load('processed_data/train_features.npy')
y_train = np.load('processed_data/train_labels.npy')
X_val = np.load('processed_data/val_features.npy')
y_val = np.load('processed_data/val_labels.npy')
X_test = np.load('processed_data/test_features.npy')
y_test = np.load('processed_data/test_labels.npy')

# Проверка и масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Аугментация данных
X_train_augmented = augment_data(X_train)
y_train_augmented = np.repeat(y_train, 3)  # Повторяем метки для увеличенных данных

# Обновляем тренировочный набор
X_train = np.vstack([X_train, X_train_augmented])
y_train = np.hstack([y_train, y_train_augmented])

# Строим модель
model = Sequential([
    Dense(512, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(8, activation='softmax')  # 8 классов эмоций
])

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# EarlyStopping с увеличением терпения
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Обучение модели
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Оценка модели
y_pred = np.argmax(model.predict(X_test), axis=-1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели на тестовых данных: {accuracy:.4f}")
print("Полный отчет классификации:")
print(classification_report(y_test, y_pred, target_names=EMOTIONS.values()))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS.values(), yticklabels=EMOTIONS.values())
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')
plt.title('Confusion Matrix')
plt.show()

# График обучения
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Тренировочная точность')
plt.plot(history.history['val_accuracy'], label='Валидационная точность')
plt.title('Точность модели на тренировочных и валидационных данных')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()
