import librosa
import numpy as np
import speech_recognition as sr

def extract_features(audio_path):
    """
    Извлекает характеристики из аудиофайла.
    """
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    pitch, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch_mean = pitch[~pitch.isna()].mean()
    return np.hstack([pitch_mean, mfcc_mean])

def recognize_speech(audio_path):
    """
    Распознаёт речь из аудиофайла и возвращает текст.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        return text
    except sr.UnknownValueError:
        return "Речь не распознана"
    except sr.RequestError:
        return "Ошибка запроса"