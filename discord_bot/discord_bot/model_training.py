import os
import numpy as np
from utils.audio_processing import extract_features
from utils.model_utils import train_model, save_model
from utils.config import AUDIO_FOLDER, MODEL_PATH

def prepare_data(audio_folder):
    """
    Подготавливает данные для обучения модели.
    """
    user_data = []
    for user_id, filename in enumerate(os.listdir(audio_folder)):
        audio_path = os.path.join(audio_folder, filename)
        features = extract_features(audio_path)
        user_data.append([user_id] + list(features))
    return np.array(user_data)

def main():
    # Подготовка данных
    data = prepare_data(AUDIO_FOLDER)
    X = data[:, 1:]  # Характеристики (тон, MFCC)
    y = data[:, 0]   # Метки (ID пользователей)

    # Нормализация данных
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Обучение модели
    model, history = train_model(X, y)

    # Сохранение модели
    save_model(model, MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")

if __name__ == "__main__":
    main()