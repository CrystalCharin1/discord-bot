import disnake
from disnake.ext import commands
import subprocess
import numpy as np
from utils.audio_processing import recognize_speech
from utils.audio_processing import extract_features
from utils.model_utils import load_model
from utils.config import KEYWORDS, MODEL_PATH

# Загрузка модели
model = load_model(MODEL_PATH)

X = np.array([
    [220.0, 1.2, 0.8, 0.5],  # Пользователь 1
    [440.0, 0.9, 1.1, 0.7],  # Пользователь 2
    [880.0, 1.5, 0.7, 0.9],  # Пользователь 3
])

# Инициализация бота
bot = commands.Bot(
    command_prefix="!",
    status=disnake.Status.online,
    activity=disnake.Game(name="В рыбалку")
)

# Функция для анализа текста
def contains_keywords(text):
    for word in KEYWORDS:
        if word in text.lower():
            return True
    return False

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
bot.run("ВАШ_ТОКЕН_БОТА")
