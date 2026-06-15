# AI Video Assistant

An AI-powered video processing tool that extracts audio from videos, transcribes speech to text, generates summaries, and enables intelligent querying using Retrieval-Augmented Generation (RAG) technology.

## Features

- **Video Processing**: Extract audio from video files
- **Speech Transcription**: Convert audio to text using advanced AI models
- **Content Summarization**: Generate concise summaries of video content
- **RAG-based Q&A**: Ask questions about video content and get context-aware answers
- **Vector Storage**: Efficient storage and retrieval of processed content using ChromaDB

## How It Works

The AI Video Assistant processes videos through a multi-stage pipeline:

1. **Video/Audio Acquisition**:
   - Downloads audio from YouTube URLs using `yt-dlp`
   - Processes and converts audio files using `pydub` and `ffmpeg-python`
   - Chunks long audio files into manageable segments for efficient processing

2. **Speech-to-Text Transcription**:
   - Uses OpenAI's Whisper model (locally hosted) for accurate speech recognition
   - Supports multiple languages and optional translation to English
   - Default model: `small` (configurable via `WHISPER_MODEL` environment variable)

3. **Content Analysis**:
   - **Summarization**: Leverages Mistral AI's language models via LangChain for generating concise summaries
   - **Extraction**: Identifies actionable items, key decisions, and questions from transcripts using Mistral
   - Text is split into chunks using LangChain's `RecursiveCharacterTextSplitter` for optimal processing

4. **Retrieval-Augmented Generation (RAG)**:
   - Embeds transcript chunks using `sentence-transformers` (HuggingFace models)
   - Stores embeddings in ChromaDB for efficient vector search
   - Retrieves relevant context for user queries
   - Generates answers using Mistral AI models with retrieved context

5. **User Interface**:
   - Built with Streamlit for an interactive web-based experience
   - Supports PDF export of summaries and transcripts using `reportlab` or `fpdf2`

**Key Technologies and Models**:
- **Whisper**: OpenAI's speech recognition model for transcription
- **Mistral AI**: Large language model for summarization, extraction, and Q&A (via `mistral-small-latest`)
- **Sentence Transformers**: For generating text embeddings (e.g., `all-MiniLM-L6-v2`)
- **ChromaDB**: Vector database for storing and retrieving embeddings
- **LangChain**: Orchestrates LLM interactions and RAG pipeline
- **PyTorch**: Backend for Whisper and audio processing

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/manoharpavuluri/ai-video-assistant.git
   cd ai-video-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or using uv:
   ```bash
   uv pip install -r requirements.txt
   ```

## Usage

### Streamlit UI

Run the interactive UI:

```bash
streamlit run streamlit_app.py --server.fileWatcherType none
```

The UI supports uploaded audio/video files, YouTube URLs, local file paths, transcript Q&A, and TXT/PDF exports.

### CLI

Place your video files in the `downloads/` directory or pass a local file path when prompted.

Run the main application:
   ```bash
   python main.py
   ```

For testing individual components:
   ```bash
   python test.py
   ```

### macOS App / DMG

Build a local `.app` bundle and `.dmg` installer:

```bash
chmod +x packaging/build_macos_dmg.sh
./packaging/build_macos_dmg.sh
```

The generated installer is written to `dist/AI Video Assistant.dmg`. The app launches the Streamlit UI using this project's `.venv`, so keep the project folder in place after installing the app.

## Project Structure

- `main.py`: Main entry point for the application
- `core/`: Core modules
  - `extractor.py`: Video/audio extraction functionality
  - `transcriber.py`: Speech-to-text transcription
  - `summarize.py`: Content summarization
  - `rag_engine.py`: RAG-based question answering
  - `vector_store.py`: Vector database operations
- `utils/`: Utility modules
  - `audio_processor.py`: Audio processing utilities
- `requirements.txt`: Python dependencies
- `test.py`: Test scripts

## Requirements

- Python 3.8+
- FFmpeg (for video processing)
- API keys for AI services (configure in environment variables)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please open an issue on GitHub.
