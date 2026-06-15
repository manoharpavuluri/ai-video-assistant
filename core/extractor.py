# actionableitems, decision, questions, summary

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_llm():
    return ChatMistralAI(model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"), temperature=0.7)

def build_chain(system_prompt: str):
    llm = get_llm()
    return (RunnablePassthrough() | RunnableLambda(lambda x : {"text": x}) | ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ]) | llm | StrOutputParser())

def extract_actionable_items(transcript: str) -> str:
    system_prompt = "You are a helpful assistant that extracts actionable items from video transcripts. "
    system_prompt += "Identify clear tasks or actions mentioned in the transcript and list them as actionable items."
    chain = build_chain(system_prompt)
    return chain.invoke(transcript)

def extract_key_decisions(transcript: str) -> str:
    system_prompt = "You are a helpful assistant that extracts key decisions from video transcripts. "
    system_prompt += "Identify important decisions or conclusions mentioned in the transcript and list them as key decisions."
    chain = build_chain(system_prompt)
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    system_prompt = (
        "You are a helpful assistant that extracts questions and uncertainties from video transcripts. "
        "Return only questions that are discussed or raised in the transcript. "
        "Write a concise numbered list of questions. Do not group multiple questions under one item."
    )
    chain = build_chain(system_prompt)
    return chain.invoke(transcript)


def answer_questions_from_transcript(transcript: str, questions: str) -> str:
    system_prompt = (
        "You answer extracted open questions using only the supplied transcript. "
        "For every question provided, return a numbered item with exactly two lines: "
        "Question: <question> and Answer: <concise answer from the transcript>. "
        "If the transcript does not answer a question, write: Answer: Not answered in the transcript. "
        "Do not create new questions. Do not omit any provided questions. "
        "Do not group subquestions together."
    )
    chain = build_chain(system_prompt)
    prompt_input = f"Transcript:\n{transcript}\n\nQuestions to answer:\n{questions}"
    return chain.invoke(prompt_input)


def extract_answered_questions(transcript: str) -> str:
    questions = extract_questions(transcript)
    if "Answer:" in questions:
        return questions
    return answer_questions_from_transcript(transcript, questions)

