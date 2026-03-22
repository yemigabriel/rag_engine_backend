from os import system
from typing import List
from openai import OpenAI


class OpenAILLMService:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def generate_answer(self, question: str, context: List[str]) -> str:
        context_str = "\n\n".join(context)
        system_prompt = f"""
            You are a helpful assistant answering questions based on a provided document.
        """
        prompt = f"""
            Use ONLY the context below to answer the question.
            If the answer is not in the context, say "I don't know based on the provided document."

            Context:
            {context_str}

            Question:
            {question}

            Answer:
            """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()