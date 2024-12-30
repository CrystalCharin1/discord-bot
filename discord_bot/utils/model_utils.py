from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

def create_model(input_shape, num_classes):
    """
    Создаёт модель нейронной сети.
    """
    model = Sequential([
        Dense(128, input_shape=input_shape, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X, y, epochs=50, validation_split=0.2):
    """
    Обучает модель на данных.
    """
    model = create_model((X.shape[1],), len(np.unique(y)))
    history = model.fit(X, y, epochs=epochs, validation_split=validation_split)
    return model, history

def save_model(model, path):
    """
    Сохраняет модель в файл.
    """
    model.save(path)

def load_model(path):
    """
    Загружает модель из файла.
    """
    return load_model(path)