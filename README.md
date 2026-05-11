# AI Video Assistant

An AI-powered video processing tool that extracts audio from videos, transcribes speech to text, generates summaries, and enables intelligent querying using Retrieval-Augmented Generation (RAG) technology.

## Features

- **Video Processing**: Extract audio from video files
- **Speech Transcription**: Convert audio to text using advanced AI models
- **Content Summarization**: Generate concise summaries of video content
- **RAG-based Q&A**: Ask questions about video content and get context-aware answers
- **Vector Storage**: Efficient storage and retrieval of processed content using ChromaDB

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

1. Place your video files in the `downloads/` directory or modify the code to point to your video location.

2. Run the main application:
   ```bash
   python main.py
   ```

3. For testing individual components:
   ```bash
   python test.py
   ```

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