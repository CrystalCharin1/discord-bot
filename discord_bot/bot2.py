import disnake
from disnake.ext import commands
import subprocess
import speech_recognition as sr
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Инициализация бота
bot = commands.Bot(command_prefix="!")

# Ключевые слова
KEYWORDS = ["мат", "оскорбление", "запрещенное слово"]

# Функция для распознавания речи
def recognize_speech(audio_path):
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

# Функция для анализа текста
def contains_keywords(text):
    for word in KEYWORDS:
        if word in text.lower():
            return True
    return False

# Функция для извлечения характеристик голоса
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    pitch, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch_mean = pitch[~pitch.isna()].mean()
    return np.hstack([pitch_mean, mfcc_mean])

# Загрузка модели
model = load_model("voice_recognition_model.h5")

# Функция для классификации пользователя
def classify_user(features):
    features = (features - X.mean(axis=0)) / X.std(axis=0)
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    return np.argmax(prediction)

# Команда для проверки
@bot.command()
async def check(ctx):
    if ctx.author.voice:
        # Запись аудио
        process = subprocess.Popen(
            ["ffmpeg", "-i", "pipe:0", "-f", "wav", "user_audio.wav"],
            stdin=subprocess.PIPE
        )
        voice_client = await ctx.author.voice.channel.connect()
        voice_client.listen(process.stdin)
        await ctx.send("Запись начата!")
        
        # Остановка записи
        voice_client.stop()
        await voice_client.disconnect()
        
        # Распознавание речи
        text = recognize_speech("user_audio.wav")
        
        # Анализ текста
        if contains_keywords(text):
            await ctx.send(f"{ctx.author.mention}, вы использовали запрещённое слово!")
            
            # Извлечение характеристик голоса
            features = extract_features("user_audio.wav")
            
            # Классификация пользователя
            user_id = classify_user(features)
            await ctx.send(f"Распознан пользователь с ID: {user_id}")
            
            # Действие бота (например, мут)
            await ctx.author.edit(mute=True)
        else:
            await ctx.send("Запрещённых слов не обнаружено.")
    else:
        await ctx.send("Вы не в голосовом канале!")

# Запуск бота
bot.run("")