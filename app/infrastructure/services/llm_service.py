from typing import List
from langchain import messages
from openai import OpenAI
from pypika import Query

from app.domain.entities import QueryRewrite
SYSTEM_PROMPT = "You are a helpful assistant answering questions based on a provided document."
USER_PROMPT_PREFIX = f"""
            You are a helpful assistant answering questions based on a provided document.

            Use the context below to answer the question.

            You may summarize or infer the answer if it is clearly supported by the context.

            If the answer truly cannot be derived from the context, say:
            "I don't know based on the provided document."
            """


class OpenAILLMService:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_answer(self, question: str, context: List[str], history: List[dict]) -> str:
        messages = self._build_messages(question, context, history)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2
        )

        return response.choices[0].message.content.strip()
    
    def generate_conversational_answer(self, question: str, context: List[str], history: List[dict]) -> str:
        messages = self._build_conversational_messages(question, context, history)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    
    def generate_answer_stream(self, question: str, context: List[str], history: List[dict]):
        messages = self._build_messages(question, context, history)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:   
                yield chunk.choices[0].delta.content
    
    def generate_conversational_answer_stream(self, question: str, context: List[str], history: List[dict]):
        messages = self._build_conversational_messages(question, context, history)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:   
                yield chunk.choices[0].delta.content
    
    def rewrite_question(self, question: str, history: List[dict]) -> QueryRewrite:
        
        conversation = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in history
        )
        system_prompt = "You are an expert query analysis engine."

        prompt = f"""
        Analyze the conversation history and the user's latest follow-up question. 

        ### CRITICAL RULES:
        1. **Routing Logic:** Set "needs_retrieval" to true if the question requires searching a database for facts. Set it to false if the question is purely conversational, a greeting, or asks you to verify a previous statement (e.g., "Are you sure?").
        2. **No Vague References:** If "needs_retrieval" is true, evaluate if the user's question contains pronouns (it, they, this) or generic pointers ("the document"). If it does, you MUST rewrite it using the exact, specific entities from the conversation history. Imagine the query is going to a dumb search engine with no memory.
        3. **The Pass-Through Rule:** If "needs_retrieval" is true, BUT the user's question is already completely standalone, explicit, and contains no vague references (e.g., a brand new topic), do NOT rewrite it. Just return the exact user question as the "standalone_query".
        4. **First-Turn Summarization:** If the user asks a generic question like "What is this document?", "Summarize this", or "What is this about?" (especially when there is no prior conversation history), you MUST rewrite it into a highly descriptive, broad search query like: "Provide a comprehensive summary of the main topics, abstract, and purpose of the document."

        ### EXAMPLES:

        Conversation:
        AI: Apple released the Vision Pro in 2024. It features spatial computing.
        User: How much does it cost?
        Output:
        {{
        "needs_retrieval": true,
        "standalone_query": "How much does the Apple Vision Pro cost?"
        }}

        Conversation:
        AI: The report covers the evaluation of LIME and SP-LIME machine learning models on the 20 newsgroups dataset.
        User: Are you sure about that information?
        Output:
        {{
        "needs_retrieval": false,
        "standalone_query": "Are you sure about that information?"
        }}

        Conversation:
        AI: The report covers the evaluation of LIME and SP-LIME machine learning models on the 20 newsgroups dataset.
        User: What is the capital of France?
        Output:
        {{
        "needs_retrieval": true,
        "standalone_query": "What is the capital of France?"
        }}
        
        Conversation:
        [Empty History]
        User: What is this document?
        Output:
        {{
        "needs_retrieval": true,
        "standalone_query": "Provide a comprehensive summary of the main topics, abstract, and purpose of the document."
        }}

        ### YOUR TURN:

        Conversation history:
        {conversation}

        Latest follow-up question:
        {question}
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            temperature=0,
            response_format=QueryRewrite
        )

        return response.choices[0].message.parsed
    
    def _build_messages(self, question: str, context: List[str], history: List[dict]) -> List[dict]:
        context_str = "\n\n".join(context)
        system_prompt = SYSTEM_PROMPT
        prompt = f"""{USER_PROMPT_PREFIX}

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
        return messages
    
    def _build_conversational_messages(self, question: str, context: List[str], history: List[dict]) -> List[dict]:
        chat_system_prompt = """
        You are a conversational assistant. Your job is to continue the conversation or clarify previous statements using ONLY the provided 'Chat History'.

        ANTI-HALLUCINATION RULES:
        1. Grounding: Your only source of truth is the 'Chat History'. Do not introduce new external facts, names, or numbers that were not already mentioned in previous messages.
        2. Verification ("Are you sure?"): If the user asks you to verify or double-check a previous statement, review the logic and claims you made in the 'Chat History'. Explain your reasoning based on what was previously discussed. 
        3. The Fallback: If the user asks a factual question that requires new information not present in the 'Chat History', you must state that you need to search your database to confirm. Do not guess.
        """

        messages = [{"role": "system", "content": chat_system_prompt}]
        
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": question})
        return messages