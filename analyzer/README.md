# Voice Analyzer

Определяет по аудиозаписи: живой голос или синтезированный.

---

## Модели

Положи в папку `models/` три файла:

- `silero_vad.onnx` — детекция речи (VAD)
- `spectra-aasist3.onnx` — детекция синтеза (AASIST)
- `voxceleb_ECAPA512_LM.onnx` — эмбеддинги голоса

---

## Установка

Требуется **Python 3.11**.

```bash
# создать окружение
python3.11 -m venv .venv

# активировать (Linux/macOS)
source .venv/bin/activate

# активировать (Windows)
.venv\Scripts\activate

# поставить зависимости
pip install -r requirements.txt
```

---

## Запуск

```bash
python main.py audio/файл.wav
```


## Что пока не работает

- **Определение пола** — заглушка `"unknown"`.