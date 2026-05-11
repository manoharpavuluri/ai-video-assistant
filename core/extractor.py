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
    system_prompt = "You are a helpful assistant that extracts questions from video transcripts. "
    system_prompt += "Identify any questions or uncertainties mentioned in the transcript and list them as questions."
    chain = build_chain(system_prompt)
    return chain.invoke(transcript)     

