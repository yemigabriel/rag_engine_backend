from email import message
from os import system
from typing import List
from click import prompt
from openai import OpenAI

MODEL_NAME = "gpt-4o-mini"

class OpenAILLMService:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def generate_answer(self, question: str, context: List[str], history: List[dict]) -> str:
        context_str = "\n\n".join(context)
        system_prompt = "You are a helpful assistant answering questions based on a provided document."
        prompt = f"""
            You are a helpful assistant answering questions based on a provided document.

            Use the context below to answer the question.

            You may summarize or infer the answer if it is clearly supported by the context.

            If the answer truly cannot be derived from the context, say:
            "I don't know based on the provided document."
            
            Context:
            {context_str}

            Question:
            {question}

            Answer:
        """
        messages = [
            {"role": "system", "content": system_prompt},
            *(history or []),
            {"role": "user", "content": prompt}
        ]
        

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2
        )

        return response.choices[0].message.content.strip()
    
    def rewrite_question(self, question: str, history: List[dict]) -> str:
        if not history:
            return question
        conversation = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in history
        )
        
        system_prompt = "You rewrite follow-up questions into clear standalone questions."
        prompt = f"""
            Given the conversation below, rewrite the latest question into a standalone question.
            
            If the question is already standalone, return it unchanged.
            
            Make the rewritten question explicit and fully self-contained.

        Conversation:
        {conversation}
        
        Follow-up question:
        {question}

        Rewritten standalone question:
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0
        )

        return response.choices[0].message.content.strip()