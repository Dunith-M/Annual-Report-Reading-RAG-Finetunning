from __future__ import annotations

import os
from typing import List, Dict, Any
from openai import OpenAI


class GroundedAnswerGenerator:
    """
    Generates answers using only retrieved annual report context.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_context_chars: int = 6000,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_context_chars = max_context_chars
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        context_blocks = []

        current_length = 0

        for chunk in chunks:
            block = (
                f"[Source: {chunk.get('chunk_id')} | "
                f"Page: {chunk.get('page_start')}]\n"
                f"{chunk.get('text')}\n"
            )

            if current_length + len(block) > self.max_context_chars:
                break

            context_blocks.append(block)
            current_length += len(block)

        return "\n---\n".join(context_blocks)

    def generate(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
    ) -> str:
        context = self.build_context(chunks)

        system_prompt = """
You are The Librarian, a strict financial annual report QA assistant.

Rules:
1. Answer only using the provided context.
2. Do not invent facts.
3. If the answer is not in the context, say: "The annual report context provided does not contain enough evidence to answer this."
4. Keep the answer concise and factual.
5. Mention page evidence naturally when useful.
"""

        user_prompt = f"""
Question:
{question}

Context:
{context}

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
        )

        return response.choices[0].message.content.strip()