import whisper
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None

def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        _model = whisper.load_model(WHISPER_MODEL)
    return _model

def transcribe_audio(file_path: str, translate: bool = False) -> str:
    model = load_model()
    print(f"Transcribing audio file: {file_path}...")
    task = "translate" if translate else "transcribe"

    result = model.transcribe(
        file_path,
        task=task,
        fp16=False
    )
    return result["text"]

def transcribe_all(chunks: list, translate: bool = False) -> str:
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}: {chunk}...")
        transcript = transcribe_audio(chunk, translate=translate)
        full_transcript += transcript + " "
    return full_transcript.strip()
