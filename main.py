from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from utils.audio_processor import process_input
from utils.youtube_transcript import fetch_youtube_transcript
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_actionable_items, extract_key_decisions, extract_answered_questions
from core.rag_engine import build_rag_chain, ask_question


def _notify(progress_callback, message: str) -> None:
    print(message)
    if progress_callback:
        progress_callback(message)


def run_pipeline(
    source: str,
    language: str = "english",
    translate: bool = False,
    include_notes: bool = True,
    progress_callback=None,
    transcriber_backend: str = "youtube-captions",
    whisper_model: str = "small",
    whisper_compute_type: str = "int8",
    llm_provider: str = "mistral",
    llm_model: str = "mistral-small-latest",
    embedding_model: str = "mistral-embed",
    allow_local_fallback: bool = False,
) -> dict:
    _notify(progress_callback, "Starting AI Video Assistant")

    transcript = None
    if transcriber_backend == "youtube-captions" and source.startswith("http"):
        _notify(progress_callback, "Fetching YouTube captions online")
        transcript = fetch_youtube_transcript(source)
        if not transcript and not allow_local_fallback:
            raise RuntimeError(
                "No YouTube captions were found. Enable local fallback or choose a local transcription model to transcribe audio."
            )

    if transcript is None:
        _notify(progress_callback, "Downloading/extracting audio and preparing chunks")
        chunks = process_input(source)

        _notify(progress_callback, f"Transcribing {len(chunks)} audio chunk(s) with {transcriber_backend}: {whisper_model}")
        transcript = transcribe_all(
            chunks,
            translate=translate,
            backend=transcriber_backend,
            model_name=whisper_model,
            compute_type=whisper_compute_type,
        )

    print(f"raw transcription (first 300 characters ) {transcript[:300]}")

    _notify(progress_callback, "Building retrieval index for transcript chat")
    rag_chain = build_rag_chain(
        transcript,
        llm_provider=llm_provider,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )

    title = "Video Analysis"
    summary = "Summary generation was skipped."
    action_item = "Action item extraction was skipped."
    decisions = "Decision extraction was skipped."
    questions = "Question extraction was skipped."

    if include_notes:
        _notify(progress_callback, "Generating title")
        title = generate_title(transcript)

        _notify(progress_callback, "Generating summary")
        summary = summarize(transcript)

        _notify(progress_callback, "Extracting action items")
        action_item = extract_actionable_items(transcript)

        _notify(progress_callback, "Extracting key decisions")
        decisions = extract_key_decisions(transcript)

        _notify(progress_callback, "Extracting and answering open questions")
        questions = extract_answered_questions(transcript)

    _notify(progress_callback, "Analysis complete")

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/English): ").strip() or "english"
    translate = input("Translate to English? (y/N): ").strip().lower() in {"y", "yes"}
    include_notes = input("Generate summary and notes? (y/N): ").strip().lower() in {"y", "yes"}
    result = run_pipeline(source, language, translate=translate, include_notes=include_notes)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")
    print("=" * 60)

    print("\nChat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\nAssistant: {answer}\n")
