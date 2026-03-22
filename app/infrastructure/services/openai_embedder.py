from typing import List

from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]
    