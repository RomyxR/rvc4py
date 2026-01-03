import os
import traceback
from io import BytesIO

import av
import librosa
import numpy as np


def wav2(i, o, format):
    inp = av.open(i, "r")
    if format == "m4a":
        format = "mp4"
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def audio2(i, o, format, sr):
    inp = av.open(i, "r")
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "f32le":
        format = "pcm_f32le"

    ostream = out.add_stream(format, channels=1)
    ostream.sample_rate = sr

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    out.close()
    inp.close()


def load_audio(file, sr):
    try:
        # Используем librosa для загрузки
        # mono=True гарантирует, что звук будет один канал (нужно для RVC)
        # sr=16000 делает ресемплинг (преобразование частоты)
        y, _ = librosa.load(file, sr=sr, mono=True)
        return y
    except Exception as e:
        print(f"Ошибка при загрузке аудио через librosa: {e}")
        raise RuntimeError(f"Не удалось загрузить файл {file}")
