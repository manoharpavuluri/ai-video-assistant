import yt_dlp
from pydub import AudioSegment
import os
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
        filename = ydl.prepare_filename(info_dict).replace(".webm", ".wav").replace(".m4a", ".wav")
        

    return filename

def convert_to_wav(mp3_path: str) -> str:
    """Convert the downloaded MP3 file to WAV format."""
    wav_path = mp3_path.replace(".mp3", ".wav")
    wav_path = os.path.splitext(mp3_path)[0] + "_con.wav"
    audio = AudioSegment.from_mp3(mp3_path)
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
        audio_path = convert_to_wav(source)
    
    print(f"Audio file ready at: {audio_path}")
    print(f"Audio file ready at: {audio_path} and chunking into {len(chunk_audio(audio_path))}-minute segments...")
    

    return chunk_audio(audio_path)
    
data = download_youtube_audio("https://www.youtube.com/watch?v=NF2aRqIlYNE")

data_final = (convert_to_wav(data))

print(chunk_audio(data_final, 10))