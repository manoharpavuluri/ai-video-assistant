import os

DEFAULT_BACKEND = os.getenv("TRANSCRIBER_BACKEND", "faster-whisper")
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "small")
DEFAULT_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

_models = {}


def _model_key(backend: str, model_name: str, compute_type: str) -> tuple[str, str, str]:
    return backend, model_name, compute_type


def load_model(
    backend: str = DEFAULT_BACKEND,
    model_name: str = DEFAULT_MODEL,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
):
    key = _model_key(backend, model_name, compute_type)
    if key in _models:
        return _models[key]

    if backend == "faster-whisper":
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            ) from exc

        print(f"Loading faster-whisper model: {model_name} ({compute_type})...")
        model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
    elif backend == "openai-whisper":
        import whisper

        print(f"Loading OpenAI Whisper model: {model_name}...")
        model = whisper.load_model(model_name)
    else:
        raise ValueError(f"Unsupported transcriber backend: {backend}")

    _models[key] = model
    return model


def transcribe_audio(
    file_path: str,
    translate: bool = False,
    backend: str = DEFAULT_BACKEND,
    model_name: str = DEFAULT_MODEL,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
) -> str:
    model = load_model(backend=backend, model_name=model_name, compute_type=compute_type)
    print(f"Transcribing audio file: {file_path}...")
    task = "translate" if translate else "transcribe"

    if backend == "faster-whisper":
        segments, _ = model.transcribe(
            file_path,
            task=task,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    result = model.transcribe(file_path, task=task, fp16=False)
    return result["text"].strip()


def transcribe_all(
    chunks: list,
    translate: bool = False,
    backend: str = DEFAULT_BACKEND,
    model_name: str = DEFAULT_MODEL,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
) -> str:
    transcripts = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}: {chunk}...")
        transcript = transcribe_audio(
            chunk,
            translate=translate,
            backend=backend,
            model_name=model_name,
            compute_type=compute_type,
        )
        if transcript:
            transcripts.append(transcript)
    return " ".join(transcripts).strip()
