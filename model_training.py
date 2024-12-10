import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import librosa
from sklearn.utils.class_weight import compute_class_weight

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

# Аугментация данных

def augment_data(features, labels):
    augmented_features, augmented_labels = [], []
    for feature, label in zip(features, labels):
        augmented_features.append(feature)
        augmented_labels.append(label)

        # Добавление шума
        noise = np.random.normal(0, 0.05, feature.shape)
        augmented_features.append(feature + noise)
        augmented_labels.append(label)

        # Изменение скорости
        try:
            stretched = librosa.effects.time_stretch(feature, rate=1.2)
            padded = np.pad(stretched, (0, max(0, len(feature) - len(stretched))), mode='constant')
            augmented_features.append(padded[:len(feature)])
            augmented_labels.append(label)
        except Exception:
            pass

        # Изменение высоты тона
        try:
            pitched = librosa.effects.pitch_shift(feature, sr=48000, n_steps=np.random.uniform(-3, 3))
            padded = np.pad(pitched, (0, max(0, len(feature) - len(pitched))), mode='constant')
            augmented_features.append(padded[:len(feature)])
            augmented_labels.append(label)
        except Exception:
            pass

        # Увеличение для нейтральной эмоции
        if label == 0:
            augmented_features.extend([feature + np.random.normal(0, 0.1, feature.shape) for _ in range(3)])
            augmented_labels.extend([label] * 3)

    return np.array(augmented_features), np.array(augmented_labels)

# Загрузка данных
X_train = np.load('processed_data/train_features.npy')
y_train = np.load('processed_data/train_labels.npy')
X_val = np.load('processed_data/val_features.npy')
y_val = np.load('processed_data/val_labels.npy')
X_test = np.load('processed_data/test_features.npy')
y_test = np.load('processed_data/test_labels.npy')

# Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Аугментация данных
X_train_augmented, y_train_augmented = augment_data(X_train, y_train)
X_train = np.vstack([X_train, X_train_augmented])
y_train = np.hstack([y_train, y_train_augmented])

# Вычисление весов классов
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weight for i, weight in enumerate(class_weights)}

# Residual Block

def residual_block(x, units):
    shortcut = x
    x = Dense(units, activation='relu', kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(units, activation='relu', kernel_regularizer='l2')(x)

    # Выравнивание размеров, если они различаются
    if shortcut.shape[-1] != units:
        shortcut = Dense(units, kernel_regularizer='l2')(shortcut)

    x = Add()([x, shortcut])
    x = BatchNormalization()(x)
    return x

# Создание модели с Residual Connections

def build_model(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(1024, activation='relu', kernel_regularizer='l2')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = residual_block(x, 512)
    x = residual_block(x, 256)

    x = Dense(128, activation='relu', kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(8, activation='softmax')(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Обучение модели
model = build_model(X_train.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights
)

# Оценка модели
y_pred = np.argmax(model.predict(X_test), axis=-1)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Точность модели на тестовых данных: {accuracy:.4f}")
print(f"F1-метрика: {f1:.4f}")
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
