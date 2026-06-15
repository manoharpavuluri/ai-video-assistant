import yt_dlp
from pydub import AudioSegment
import os
from pathlib import Path

download_dir = "downloads"
chunk_dir = "chunks"

os.makedirs(download_dir, exist_ok=True)
os.makedirs(chunk_dir, exist_ok=True)

def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(download_dir, "%(title)s.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = _resolve_downloaded_audio_path(ydl, info_dict)

    return filename


def _resolve_downloaded_audio_path(ydl: yt_dlp.YoutubeDL, info_dict: dict) -> str:
    requested_downloads = info_dict.get("requested_downloads") or []
    for download in requested_downloads:
        filepath = download.get("filepath")
        if filepath and os.path.exists(filepath):
            return filepath

    prepared_path = Path(ydl.prepare_filename(info_dict))
    wav_path = prepared_path.with_suffix(".wav")
    if wav_path.exists():
        return str(wav_path)
    if prepared_path.exists():
        return str(prepared_path)

    raise FileNotFoundError(f"Could not find downloaded audio for: {info_dict.get('title', 'video')}")


def convert_to_wav(mp3_path: str) -> str:
    """Convert an audio/video file to a mono 16kHz WAV file."""
    wav_path = os.path.splitext(mp3_path)[0] + "_con.wav"
    audio = AudioSegment.from_file(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(wav_path, format="wav")
    return wav_path

def chunk_audio(wav_path: str, chunk_length_min: int = 10) -> list:
    """Chunk the WAV file into smaller segments."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_length_min * 60 * 1000
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def process_input(source: str) -> list:
    """Process the input source (YouTube URL or local file path) and return a list of audio chunks."""
    if source.startswith("http"):
        audio_path = download_youtube_audio(source)
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Input file not found: {source}")
        audio_path = convert_to_wav(source)
    
    print(f"Audio file ready at: {audio_path}")
    chunks = chunk_audio(audio_path)
    print(f"Chunked audio into {len(chunks)} segment(s).")
    return chunks
