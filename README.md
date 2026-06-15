# AI Video Assistant

AI Video Assistant is a Streamlit app for turning video, audio, YouTube captions, or local media files into a searchable meeting/video workspace. It can transcribe speech, summarize content, extract actions, decisions, and open questions, answer transcript questions with RAG, export notes, and read generated sections aloud with local Kokoro TTS.

## Features

- Upload audio/video files, paste YouTube URLs, or use local file paths
- Prefer YouTube captions when available, with optional local transcription fallback
- Local transcription options with faster-whisper or legacy OpenAI Whisper
- Optional translation of speech to English
- Summary, action items, key decisions, and open questions
- Transcript Q&A using retrieval-augmented generation
- Local or online model choices for Q&A and embeddings
- TXT and PDF export
- Natural read-back for Summary, Actions, Decisions, and Questions
- Kokoro TTS voices with in-player speed control
- Browser speech fallback when Kokoro is not selected
- macOS `.app` and `.dmg` packaging script

## How It Works

1. **Source processing**
   - YouTube URLs can use online captions first.
   - Uploaded/local media is converted to mono 16 kHz WAV with `pydub`/FFmpeg.
   - Long audio is chunked before transcription.

2. **Transcription**
   - `faster-whisper` provides the main local transcription path.
   - Legacy OpenAI Whisper remains available as a fallback.
   - YouTube captions can avoid local transcription when available.

3. **Analysis**
   - LangChain orchestrates summarization and extraction.
   - Mistral API or local Ollama models can power Q&A depending on the UI selection.
   - The app extracts summary, action items, key decisions, and open questions.

4. **RAG chat**
   - Transcript chunks are embedded with Mistral embeddings or local HuggingFace embeddings.
   - ChromaDB stores the local vector index.
   - Questions retrieve relevant transcript chunks before answer generation.

5. **Read-back audio**
   - Kokoro TTS generates local WAV audio for Summary, Actions, Decisions, and Questions.
   - Audio is generated automatically after analysis with the default voice.
   - The visible markdown is preserved, while the speech text is cleaned for natural narration.
   - Playback speed is adjusted inside the audio player without restarting playback.

## Requirements

- Python 3.10+
- FFmpeg available on your PATH
- Optional: `MISTRAL_API_KEY` for Mistral-powered online Q&A and embeddings
- Optional: Ollama with selected local models for local Q&A
- Internet access the first time Kokoro downloads its model files from Hugging Face

On macOS, FFmpeg can be installed with:

```bash
brew install ffmpeg
```

## Installation

```bash
git clone git@github.com:manoharpavuluri/ai-video-assistant.git
cd ai-video-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file if you want online Mistral features:

```bash
MISTRAL_API_KEY=your_key_here
```

## Run The App

```bash
source .venv/bin/activate
streamlit run streamlit_app.py --server.fileWatcherType none
```

Then open the Streamlit URL shown in the terminal.

## Read-Back Audio

The app has two read-back engines:

- **Kokoro TTS**: default, local, higher quality, multiple voices
- **Browser voice**: free browser-native fallback using the Web Speech API

Kokoro audio is generated automatically after analysis for:

- Summary
- Actions
- Decisions
- Questions

Use the audio player's speed slider to change playback speed while listening. Changing speed does not regenerate audio or restart playback. Changing the voice or regenerating analysis does require fresh audio.

## macOS App / DMG

Build the local `.app` bundle and `.dmg` installer:

```bash
./packaging/build_macos_dmg.sh
```

The installer is written to:

```text
dist/AI Video Assistant.dmg
```

The packaged app launches this project's Streamlit app through the project `.venv`, so keep the project folder and virtual environment available after installing.

## Project Structure

```text
streamlit_app.py              Streamlit UI, exports, and read-back controls
main.py                       Pipeline entry point
core/
  rag_engine.py               Transcript Q&A and model selection
  summarize.py                Summary generation helpers
  transcriber.py              Caption and local transcription paths
  vector_store.py             Chroma/vector index helpers
utils/
  audio_processor.py          Media download, conversion, and chunking
  youtube_transcript.py       YouTube caption retrieval
assets/                       App icons
packaging/build_macos_dmg.sh  macOS app/DMG builder
dist/                         Built app and DMG artifacts
```

## Notes

- `.env`, `.venv`, caches, downloads, ChromaDB data, and generated transcript chunks are intentionally ignored.
- `dist/` is tracked in this repository because the packaged app/DMG is part of the project deliverable.
- The first Kokoro use may take longer because model files are downloaded and cached locally.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
