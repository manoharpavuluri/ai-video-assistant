from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from core.rag_engine import ask_question
from main import run_pipeline


load_dotenv()

SUPPORTED_UPLOADS = ["mp3", "mp4", "m4a", "mov", "wav", "webm"]

TRANSCRIPTION_PRESETS = {
    "Online - YouTube captions": {"backend": "youtube-captions", "model": "captions", "compute_type": "none"},
    "Local fallback - small int8": {"backend": "faster-whisper", "model": "small", "compute_type": "int8"},
    "Local fallback - medium int8": {"backend": "faster-whisper", "model": "medium", "compute_type": "int8"},
    "Local fallback - distil-large-v3": {"backend": "faster-whisper", "model": "distil-large-v3", "compute_type": "int8"},
    "Legacy local - OpenAI Whisper small": {"backend": "openai-whisper", "model": "small", "compute_type": "int8"},
}

LLM_OPTIONS = {
    "Online - Mistral API small": {"provider": "mistral", "model": "mistral-small-latest"},
    "Local - Qwen3 4B": {"provider": "ollama", "model": "qwen3:4b"},
    "Local - Ollama llama3.2": {"provider": "ollama", "model": "llama3.2:latest"},
    "Local - Ollama gpt-oss 20b": {"provider": "ollama", "model": "gpt-oss:20b"},
}

EMBEDDING_OPTIONS = {
    "Online - Mistral embeddings": "mistral-embed",
    "Local - MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "Local - BGE small": "BAAI/bge-small-en-v1.5",
}

READBACK_ENGINES = ["Kokoro TTS", "Browser voice"]

KOKORO_VOICES = {
    "Heart - US female": "af_heart",
    "Bella - US female": "af_bella",
    "Nicole - US female": "af_nicole",
    "Sarah - US female": "af_sarah",
    "Adam - US male": "am_adam",
    "Michael - US male": "am_michael",
    "Emma - UK female": "bf_emma",
    "Isabella - UK female": "bf_isabella",
    "George - UK male": "bm_george",
    "Lewis - UK male": "bm_lewis",
}

KOKORO_LANG_CODE = "a"
MAX_READBACK_CHARS = 6000
AUTO_READBACK_SECTIONS = [
    ("summary", "Summary", "summary"),
    ("action_items", "Actions", "actions"),
    ("key_decisions", "Decisions", "decisions"),
    ("open_questions", "Questions", "questions"),
]


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def build_txt_export(result: dict) -> str:
    sections = [
        ("Title", result.get("title", "")),
        ("Summary", result.get("summary", "")),
        ("Action Items", result.get("action_items", "")),
        ("Key Decisions", result.get("key_decisions", "")),
        ("Open Questions", result.get("open_questions", "")),
        ("Transcript", result.get("transcript", "")),
    ]
    return "\n\n".join(f"{heading}\n{'=' * len(heading)}\n{body}" for heading, body in sections)


def build_pdf_styles() -> dict:
    base_styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocumentTitle",
            parent=base_styles["Title"],
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#2f3140"),
            spaceAfter=18,
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#2f3140"),
            spaceBefore=14,
            spaceAfter=8,
        ),
        "heading": ParagraphStyle(
            "MarkdownHeading",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#2f3140"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "subheading": ParagraphStyle(
            "MarkdownSubheading",
            parent=base_styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#2f3140"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "ExportBody",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=colors.HexColor("#31333f"),
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "ExportBullet",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            leftIndent=18,
            firstLineIndent=0,
            bulletIndent=6,
            textColor=colors.HexColor("#31333f"),
            spaceAfter=4,
        ),
    }


def markdown_to_pdf_markup(text: str) -> str:
    markup = escape(str(text or "").strip())
    markup = re.sub(r"`([^`]*)`", r"<font name='Courier'>\1</font>", markup)
    markup = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", markup)
    markup = re.sub(r"__([^_]+)__", r"<b>\1</b>", markup)
    markup = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", markup)
    markup = re.sub(r"_([^_]+)_", r"<i>\1</i>", markup)
    markup = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", markup)
    markup = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", markup)
    return markup or " "


def append_markdown_pdf(story: list, body: str, styles: dict) -> None:
    previous_blank = False
    for raw_line in str(body or "").splitlines():
        if not raw_line.strip():
            if not previous_blank:
                story.append(Spacer(1, 5))
            previous_blank = True
            continue

        previous_blank = False
        stripped = raw_line.strip()
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        bullet_match = re.match(r"^(\s*)([-*+]|[•◦])\s+(.+)$", raw_line)
        number_match = re.match(r"^(\s*)(\d+)[.)]\s+(.+)$", raw_line)

        if heading_match:
            level = len(heading_match.group(1))
            style = styles["heading"] if level <= 2 else styles["subheading"]
            story.append(Paragraph(markdown_to_pdf_markup(heading_match.group(2)), style))
        elif bullet_match:
            indent = len(bullet_match.group(1).replace("\t", "    "))
            level = max(0, min(3, indent // 2))
            bullet_style = ParagraphStyle(
                f"ExportBullet{level}",
                parent=styles["bullet"],
                leftIndent=18 + (level * 18),
                bulletIndent=6 + (level * 18),
            )
            bullet = "•" if level == 0 else "-"
            story.append(Paragraph(markdown_to_pdf_markup(bullet_match.group(3)), bullet_style, bulletText=bullet))
        elif number_match:
            indent = len(number_match.group(1).replace("\t", "    "))
            level = max(0, min(3, indent // 2))
            number_style = ParagraphStyle(
                f"ExportNumber{level}",
                parent=styles["bullet"],
                leftIndent=20 + (level * 18),
                bulletIndent=3 + (level * 18),
            )
            story.append(
                Paragraph(
                    markdown_to_pdf_markup(number_match.group(3)),
                    number_style,
                    bulletText=f"{number_match.group(2)}.",
                )
            )
        else:
            story.append(Paragraph(markdown_to_pdf_markup(stripped), styles["body"]))


def draw_pdf_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#8a8f98"))
    canvas.drawString(0.75 * inch, 0.45 * inch, "AI Video Assistant")
    canvas.drawRightString(7.75 * inch, 0.45 * inch, f"Page {document.page}")
    canvas.restoreState()


def build_pdf_export(result: dict) -> bytes:
    buffer = io.BytesIO()
    title = result.get("title") or "AI Video Assistant"
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        title=title,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.68 * inch,
        bottomMargin=0.68 * inch,
    )
    styles = build_pdf_styles()
    story = [Paragraph(markdown_to_pdf_markup(title), styles["title"])]

    for heading, body in [
        ("Summary", result.get("summary", "")),
        ("Action Items", result.get("action_items", "")),
        ("Key Decisions", result.get("key_decisions", "")),
        ("Open Questions", result.get("open_questions", "")),
        ("Transcript", result.get("transcript", "")),
    ]:
        if not str(body or "").strip():
            continue
        story.append(Paragraph(heading, styles["section"]))
        append_markdown_pdf(story, body, styles)
        story.append(Spacer(1, 10))

    document.build(story, onFirstPage=draw_pdf_footer, onLaterPages=draw_pdf_footer)
    return buffer.getvalue()


def clear_readback_audio() -> None:
    for key in list(st.session_state):
        if key.startswith("readback_audio_") or key.startswith("readback_signature_"):
            st.session_state.pop(key, None)


def reset_results() -> None:
    for key in ("result", "source_path", "chat_history", "readback_result_signature"):
        st.session_state.pop(key, None)
    clear_readback_audio()


def prepare_text_for_speech(text: str) -> str:
    speech_text = str(text or "")
    speech_text = re.sub(r"```.*?```", " ", speech_text, flags=re.DOTALL)
    speech_text = re.sub(r"`([^`]*)`", r"\1", speech_text)
    speech_text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", speech_text)
    speech_text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", speech_text)
    speech_text = re.sub(r"^\s{0,3}#{1,6}\s+", "", speech_text, flags=re.MULTILINE)
    speech_text = re.sub(r"^\s*[-*+]\s+", "", speech_text, flags=re.MULTILINE)
    speech_text = re.sub(r"^\s*\d+[.)]\s+", "", speech_text, flags=re.MULTILINE)
    speech_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", speech_text)
    speech_text = re.sub(r"__([^_]+)__", r"\1", speech_text)
    speech_text = re.sub(r"\*([^*]+)\*", r"\1", speech_text)
    speech_text = re.sub(r"_([^_]+)_", r"\1", speech_text)
    speech_text = re.sub(r"\b[A-Za-z]+://\S+", " ", speech_text)
    speech_text = speech_text.replace("&", " and ")
    speech_text = re.sub(r"(\d+)\s*[-–—]{1,2}\s*(\d+)\s*%", r"\1 to \2 percent", speech_text)
    speech_text = re.sub(r"(\d+)\s*[-–—]{1,2}\s*(\d+)", r"\1 to \2", speech_text)
    speech_text = speech_text.replace("%", " percent ")
    speech_text = speech_text.replace("/", " or ")
    speech_text = re.sub(r"[•◦▪▫]", "", speech_text)
    speech_text = re.sub(r"[-–—]{2,}", ". ", speech_text)
    speech_text = re.sub(r"\s*[-–—]\s*", ", ", speech_text)
    speech_text = re.sub(r"[“”]", '"', speech_text)
    speech_text = re.sub(r"[‘’]", "'", speech_text)
    speech_text = re.sub(r"\.{2,}", ". ", speech_text)
    speech_text = re.sub(r"!{2,}", "! ", speech_text)
    speech_text = re.sub(r"\?{2,}", "? ", speech_text)
    speech_text = re.sub(r"\s*\n\s*", ". ", speech_text)
    speech_text = re.sub(r"\s+", " ", speech_text)
    speech_text = re.sub(r"\s+([,.;:!?])", r"\1", speech_text)
    speech_text = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", speech_text)
    speech_text = re.sub(r"(?:\.\s*){2,}", ". ", speech_text)
    speech_text = re.sub(r"\s+", " ", speech_text)
    return speech_text.strip(" .,")


def get_readback_text(text: str) -> tuple[str, bool]:
    cleaned_text = prepare_text_for_speech(text)
    if len(cleaned_text) <= MAX_READBACK_CHARS:
        return cleaned_text, False
    return cleaned_text[:MAX_READBACK_CHARS].rsplit(" ", 1)[0], True


@st.cache_resource(show_spinner=False)
def get_kokoro_pipeline(lang_code: str):
    from kokoro import KPipeline

    return KPipeline(lang_code=lang_code)


@st.cache_data(show_spinner=False, max_entries=24)
def synthesize_kokoro_audio(text: str, voice: str, speed: float, lang_code: str) -> bytes:
    import numpy as np
    import soundfile as sf

    pipeline = get_kokoro_pipeline(lang_code)
    audio_chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        audio_chunks.append(np.asarray(audio))

    if not audio_chunks:
        raise RuntimeError("Kokoro did not return audio for this text.")

    audio = np.concatenate(audio_chunks)
    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format="WAV")
    return buffer.getvalue()


def default_readback_voice() -> str:
    return KOKORO_VOICES[next(iter(KOKORO_VOICES))]


def readback_signature(text: str, voice: str) -> str:
    return hashlib.sha1(f"{voice}|{text}".encode("utf-8")).hexdigest()


def result_readback_signature(result: dict) -> str:
    parts = [str(result.get(result_key, "")) for result_key, _, _ in AUTO_READBACK_SECTIONS]
    return hashlib.sha1("\n---section---\n".join(parts).encode("utf-8")).hexdigest()


def generate_default_readback_audio(result: dict, progress_callback=None) -> None:
    voice = default_readback_voice()
    for result_key, label, section_id in AUTO_READBACK_SECTIONS:
        readback_text, _ = get_readback_text(result.get(result_key, ""))
        if not readback_text:
            continue

        signature = readback_signature(readback_text, voice)
        audio_key = f"readback_audio_{section_id}"
        signature_key = f"readback_signature_{section_id}"
        if st.session_state.get(audio_key) and st.session_state.get(signature_key) == signature:
            continue

        if progress_callback:
            progress_callback(f"Generating {label.lower()} read-back audio")
        st.session_state[audio_key] = synthesize_kokoro_audio(
            readback_text,
            voice=voice,
            speed=1.0,
            lang_code=KOKORO_LANG_CODE,
        )
        st.session_state[signature_key] = signature


def render_playback_audio(audio_bytes: bytes, section_id: str) -> None:
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    player_html = f"""
    <div class="kokoro-player">
        <audio id="audio-{section_id}" controls preload="metadata">
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        </audio>
        <div class="speed-row">
            <label for="speed-{section_id}">Speed</label>
            <input id="speed-{section_id}" type="range" min="0.7" max="1.3" step="0.05" value="1">
            <output id="speed-value-{section_id}">1.00x</output>
        </div>
    </div>
    <script>
    (() => {{
        const audio = document.getElementById("audio-{section_id}");
        const speed = document.getElementById("speed-{section_id}");
        const speedValue = document.getElementById("speed-value-{section_id}");
        const setSpeed = () => {{
            const rate = Number(speed.value);
            audio.playbackRate = rate;
            audio.defaultPlaybackRate = rate;
            speedValue.textContent = `${{rate.toFixed(2)}}x`;
        }};
        speed.addEventListener("input", setSpeed);
        audio.addEventListener("play", setSpeed);
        setSpeed();
    }})();
    </script>
    <style>
    .kokoro-player {{
        width: 100%;
    }}
    .kokoro-player audio {{
        width: 100%;
    }}
    .speed-row {{
        align-items: center;
        display: grid;
        gap: 0.65rem;
        grid-template-columns: auto minmax(10rem, 1fr) 4rem;
        margin-top: 0.55rem;
        font: 0.9rem system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .speed-row input {{
        accent-color: #ff4b4b;
        width: 100%;
    }}
    .speed-row output {{
        color: rgba(49, 51, 63, 0.75);
        text-align: right;
    }}
    </style>
    """
    components.html(player_html, height=105)


def render_kokoro_readback(label: str, text: str, section_id: str) -> None:
    voice_label = st.session_state.get("readback_voice", next(iter(KOKORO_VOICES)))
    voice = KOKORO_VOICES.get(voice_label, "af_heart")
    readback_text, truncated = get_readback_text(text)
    if not readback_text:
        st.caption("Nothing to read back yet.")
        return

    signature = readback_signature(readback_text, voice)
    audio_key = f"readback_audio_{section_id}"
    signature_key = f"readback_signature_{section_id}"

    with st.container(border=True):
        st.markdown(f"**Read back {label}**")
        controls = st.columns([1.2, 2.4, 1.2])
        generate_label = "Regenerate audio" if st.session_state.get(audio_key) else "Generate audio"
        if controls[0].button(generate_label, key=f"generate_{section_id}_audio"):
            try:
                with st.spinner(f"Generating {label.lower()} audio with Kokoro..."):
                    st.session_state[audio_key] = synthesize_kokoro_audio(
                        readback_text,
                        voice=voice,
                        speed=1.0,
                        lang_code=KOKORO_LANG_CODE,
                    )
                    st.session_state[signature_key] = signature
            except ImportError as exc:
                st.error(f"Kokoro read back is not installed yet: {exc}")
                st.info("Install kokoro and soundfile, and make sure espeak-ng is available on the machine.")
            except Exception as exc:
                st.error(f"Could not generate Kokoro audio: {exc}")

        controls[1].caption(f"Voice: {voice_label} | adjust speed in the audio player")
        if controls[2].button("Clear audio", key=f"clear_{section_id}_audio"):
            st.session_state.pop(audio_key, None)
            st.session_state.pop(signature_key, None)
            st.rerun()

        if truncated:
            st.caption("Read back is limited to the first part of very long sections.")

        if st.session_state.get(audio_key) and st.session_state.get(signature_key) == signature:
            render_playback_audio(st.session_state[audio_key], section_id)
        elif st.session_state.get(audio_key):
            st.caption("Voice or text changed. Regenerate audio to hear the updated version.")


def render_readback_controls(label: str, text: str, section_id: str) -> None:
    cleaned_text, truncated = get_readback_text(text)
    if st.session_state.get("readback_engine", "Kokoro TTS") == "Kokoro TTS":
        render_kokoro_readback(label, cleaned_text, section_id)
        return

    if not cleaned_text:
        st.caption("Nothing to read back yet.")
        return

    with st.container(border=True):
        st.markdown(f"**Read back {label}**")
        if truncated:
            st.caption("Read back is limited to the first part of very long sections.")

    payload = json.dumps(cleaned_text)
    rate = float(st.session_state.get("readback_speed", 1.0))
    controls_html = f"""
    <div class="readback-controls" data-section="{section_id}">
        <button type="button" id="play-{section_id}" title="Read {label} aloud">Read</button>
        <button type="button" id="pause-{section_id}" title="Pause read back">Pause</button>
        <button type="button" id="resume-{section_id}" title="Resume read back">Resume</button>
        <button type="button" id="stop-{section_id}" title="Stop read back">Stop</button>
        <span id="status-{section_id}" aria-live="polite"></span>
    </div>
    <script>
    (() => {{
        const text = {payload};
        const status = document.getElementById("status-{section_id}");
        const setStatus = (message) => {{ status.textContent = message; }};
        const supported = "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;

        if (!supported) {{
            setStatus("Read back is not supported in this browser.");
            return;
        }}

        const speak = () => {{
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = {rate};
            utterance.pitch = 1;
            utterance.onstart = () => setStatus("Reading...");
            utterance.onpause = () => setStatus("Paused");
            utterance.onresume = () => setStatus("Reading...");
            utterance.onend = () => setStatus("Finished");
            utterance.onerror = () => setStatus("Could not read this section.");
            window.speechSynthesis.speak(utterance);
        }};

        document.getElementById("play-{section_id}").addEventListener("click", speak);
        document.getElementById("pause-{section_id}").addEventListener("click", () => {{
            window.speechSynthesis.pause();
            setStatus("Paused");
        }});
        document.getElementById("resume-{section_id}").addEventListener("click", () => {{
            window.speechSynthesis.resume();
            setStatus("Reading...");
        }});
        document.getElementById("stop-{section_id}").addEventListener("click", () => {{
            window.speechSynthesis.cancel();
            setStatus("Stopped");
        }});
    }})();
    </script>
    <style>
    .readback-controls {
        align-items: center;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0 0 0.85rem;
    }
    .readback-controls button {
        background: #ffffff;
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 6px;
        color: #31333f;
        cursor: pointer;
        font: 600 0.9rem system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        line-height: 1;
        min-height: 2.25rem;
        padding: 0.55rem 0.75rem;
    }
    .readback-controls button:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
    }
    .readback-controls span {
        color: rgba(49, 51, 63, 0.65);
        font: 0.86rem system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        min-width: 5rem;
    }
    </style>
    """
    components.html(controls_html, height=56)


st.set_page_config(
    page_title="AI Video Assistant",
    page_icon=Image.open(Path(__file__).parent / "assets/app_icon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.35rem; }
    .stDownloadButton button, .stButton button { border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Video Assistant")
st.caption("Transcribe, summarize, extract follow-ups, and ask questions about video or audio.")

with st.sidebar:
    st.header("Source")
    source_mode = st.radio("Input type", ["Upload file", "YouTube URL", "Local path"], label_visibility="collapsed")
    uploaded_file = None
    source_value = ""

    if source_mode == "Upload file":
        uploaded_file = st.file_uploader("Video or audio file", type=SUPPORTED_UPLOADS)
    elif source_mode == "YouTube URL":
        source_value = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    else:
        source_value = st.text_input("Local file path", placeholder="/Users/you/Downloads/meeting.mp4")

    translate = st.checkbox("Translate speech to English", value=False)
    include_notes = st.checkbox("Generate summary and notes", value=False, help="This makes extra LLM calls after transcript chat is ready.")
    allow_local_fallback = st.checkbox("Allow local fallback if captions are missing", value=False)

    st.divider()
    st.header("Models")
    transcription_choice = st.selectbox("Transcription", list(TRANSCRIPTION_PRESETS), index=0)
    llm_choice = st.selectbox("Q&A model", list(LLM_OPTIONS), index=0)
    embedding_choice = st.selectbox("Retrieval embeddings", list(EMBEDDING_OPTIONS), index=0)

    st.divider()
    st.header("Read back")
    st.selectbox("Voice engine", READBACK_ENGINES, index=0, key="readback_engine")
    if st.session_state.get("readback_engine") == "Kokoro TTS":
        st.selectbox("Kokoro voice", list(KOKORO_VOICES), index=0, key="readback_voice")
    else:
        st.slider("Speech speed", 0.70, 1.30, 1.00, 0.05, key="readback_speed")

    st.divider()
    st.header("Setup")
    if os.getenv("MISTRAL_API_KEY"):
        st.success("MISTRAL_API_KEY found for online mode")
    else:
        st.warning("Add MISTRAL_API_KEY to .env for online Q&A and embeddings.")
    if LLM_OPTIONS[llm_choice]["provider"] == "ollama" or EMBEDDING_OPTIONS[embedding_choice] != "mistral-embed" or TRANSCRIPTION_PRESETS[transcription_choice]["backend"] != "youtube-captions":
        st.warning("A selected option may use local CPU/GPU resources.")

    run_clicked = st.button("Analyze", type="primary", use_container_width=True)
    if st.button("Clear", use_container_width=True):
        reset_results()
        st.rerun()

if run_clicked:
    try:
        if source_mode == "Upload file":
            if uploaded_file is None:
                st.error("Choose a file first.")
                st.stop()
            source_path = save_uploaded_file(uploaded_file)
        else:
            source_path = source_value.strip()
            if not source_path:
                st.error("Enter a source first.")
                st.stop()

        st.session_state["source_path"] = source_path
        st.session_state["chat_history"] = []
        st.session_state.pop("readback_result_signature", None)
        clear_readback_audio()

        with st.status("Processing media...", expanded=True) as status:
            def report_progress(message: str) -> None:
                status.write(message)

            transcription_config = TRANSCRIPTION_PRESETS[transcription_choice]
            llm_config = LLM_OPTIONS[llm_choice]
            result_data = run_pipeline(
                source_path,
                translate=translate,
                include_notes=include_notes,
                progress_callback=report_progress,
                transcriber_backend=transcription_config["backend"],
                whisper_model=transcription_config["model"],
                whisper_compute_type=transcription_config["compute_type"],
                llm_provider=llm_config["provider"],
                llm_model=llm_config["model"],
                embedding_model=EMBEDDING_OPTIONS[embedding_choice],
                allow_local_fallback=allow_local_fallback,
            )
            st.session_state["result"] = result_data
            generate_default_readback_audio(result_data, progress_callback=report_progress)
            st.session_state["readback_result_signature"] = result_readback_signature(result_data)
            status.update(label="Analysis and read-back audio complete", state="complete", expanded=False)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

result = st.session_state.get("result")

if not result:
    left, middle, right = st.columns(3)
    left.metric("1", "Add a source")
    middle.metric("2", "Run analysis")
    right.metric("3", "Chat and export")
    st.info("Use the sidebar to upload a media file, paste a YouTube URL, or point to a local file.")
    st.stop()

st.subheader(result.get("title") or "Analysis")

overview_tabs = st.tabs(["Summary", "Actions", "Decisions", "Questions", "Transcript", "Chat"])

with overview_tabs[0]:
    summary = result.get("summary") or ""
    render_readback_controls("Summary", summary, "summary")
    st.markdown(summary or "_No summary returned._")

with overview_tabs[1]:
    action_items = result.get("action_items") or ""
    render_readback_controls("Actions", action_items, "actions")
    st.markdown(action_items or "_No action items returned._")

with overview_tabs[2]:
    key_decisions = result.get("key_decisions") or ""
    render_readback_controls("Decisions", key_decisions, "decisions")
    st.markdown(key_decisions or "_No key decisions returned._")

with overview_tabs[3]:
    open_questions = result.get("open_questions") or ""
    render_readback_controls("Questions", open_questions, "questions")
    st.markdown(open_questions or "_No open questions returned._")

with overview_tabs[4]:
    st.text_area("Transcript", result.get("transcript", ""), height=420, label_visibility="collapsed")

with overview_tabs[5]:
    st.write("Ask a question about the transcript.")
    question = st.chat_input("What did they decide?")
    if question:
        with st.spinner("Searching the transcript..."):
            answer = ask_question(result["rag_chain"], question)
        st.session_state.setdefault("chat_history", []).append((question, answer))

    for user_question, assistant_answer in st.session_state.get("chat_history", []):
        with st.chat_message("user"):
            st.write(user_question)
        with st.chat_message("assistant"):
            st.write(assistant_answer)

st.divider()
export_name = (result.get("title") or "ai-video-assistant").strip().replace("/", "-")
download_left, download_right = st.columns(2)
download_left.download_button(
    "Download TXT",
    data=build_txt_export(result),
    file_name=f"{export_name}.txt",
    mime="text/plain",
    use_container_width=True,
)
download_right.download_button(
    "Download PDF",
    data=build_pdf_export(result),
    file_name=f"{export_name}.pdf",
    mime="application/pdf",
    use_container_width=True,
)
