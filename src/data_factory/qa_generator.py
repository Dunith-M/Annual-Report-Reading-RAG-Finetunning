import os
import json
import random
from pathlib import Path
from typing import List, Literal

import jsonlines
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

QUESTION_MODEL = os.getenv("QUESTION_MODEL", "gpt-4.1-mini")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4.1-mini")


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "chunks.json"
QA_PAIRS_PATH = ROOT_DIR / "data" / "processed" / "qa_pairs.json"
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "train.jsonl"
TEST_PATH = ROOT_DIR / "data" / "processed" / "golden_test_set.jsonl"


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------

Category = Literal["hard_fact", "strategic_summary", "stylistic_creative"]


class GeneratedQuestion(BaseModel):
    question: str = Field(min_length=10)
    category: Category


class QuestionGenerationResponse(BaseModel):
    questions: List[GeneratedQuestion]


class QAPair(BaseModel):
    instruction: str
    input: str
    output: str
    source_chunk_id: str
    page_reference: str
    category: Category


# ---------------------------------------------------------
# File Helpers
# ---------------------------------------------------------

def load_chunks(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"chunks.json not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a list of chunk objects.")

    required_fields = {"chunk_id", "page_start", "page_end", "text"}

    for chunk in chunks:
        missing = required_fields - set(chunk.keys())
        if missing:
            raise ValueError(f"Chunk missing fields: {missing}")

    return chunks


def save_json(data: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(data: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(path, mode="w") as writer:
        for item in data:
            writer.write(item)


# ---------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------

def build_question_prompt(chunk_text: str) -> str:
    return f"""
You are generating synthetic fine-tuning questions from Uber's 2024 Annual Report.

Generate exactly 10 questions from the provided chunk.

You must generate a balanced set:
- 4 hard_fact questions
- 3 strategic_summary questions
- 3 stylistic_creative questions

Definitions:
1. hard_fact:
   Questions about exact facts, numbers, names, dates, segments, risks, revenue, operations, or report details.

2. strategic_summary:
   Questions requiring business interpretation based on the chunk.

3. stylistic_creative:
   Questions asking to rewrite, summarize, or explain the content in a polished business/investor-friendly tone.

Rules:
- Use only the provided chunk.
- Do not invent facts.
- Do not ask questions that cannot be answered from the chunk.
- Avoid duplicate questions.
- Return valid JSON only.
- Do not include markdown.

Required JSON format:
{{
  "questions": [
    {{
      "question": "Question text here",
      "category": "hard_fact"
    }}
  ]
}}

Chunk:
\"\"\"
{chunk_text}
\"\"\"
"""


def build_answer_prompt(question: str, category: str, chunk_text: str) -> str:
    return f"""
You are answering questions using only Uber's 2024 Annual Report chunk.

Question category: {category}

Question:
{question}

Rules:
- Answer using only the provided chunk.
- Do not use outside knowledge.
- If the chunk does not contain enough information, answer:
  "The provided chunk does not contain enough information to answer this question."
- Be accurate and concise.
- For hard_fact questions, answer directly with exact facts.
- For strategic_summary questions, explain the business meaning clearly.
- For stylistic_creative questions, write in a polished professional tone.
- Do not include markdown.

Chunk:
\"\"\"
{chunk_text}
\"\"\"
"""


# ---------------------------------------------------------
# LLM Calls
# ---------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8)
)
def generate_questions(chunk_text: str) -> QuestionGenerationResponse:
    prompt = build_question_prompt(chunk_text)

    response = client.chat.completions.create(
        model=QUESTION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a strict dataset generation assistant. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4,
        response_format={"type": "json_object"}
    )

    raw_content = response.choices[0].message.content
    parsed = json.loads(raw_content)

    return QuestionGenerationResponse.model_validate(parsed)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8)
)
def generate_answer(question: str, category: str, chunk_text: str) -> str:
    prompt = build_answer_prompt(question, category, chunk_text)

    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You answer strictly from the provided source text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()
    return answer


# ---------------------------------------------------------
# Main Dataset Generation Logic
# ---------------------------------------------------------

def build_page_reference(page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return f"page {page_start}"
    return f"pages {page_start}-{page_end}"


def generate_qa_pairs(chunks: List[dict]) -> List[dict]:
    qa_pairs = []
    seen_questions = set()

    for chunk in tqdm(chunks, desc="Generating Q/A pairs"):
        chunk_id = chunk["chunk_id"]
        chunk_text = chunk["text"]
        page_reference = build_page_reference(
            chunk["page_start"],
            chunk["page_end"]
        )

        if not chunk_text or len(chunk_text.strip()) < 200:
            continue

        try:
            question_response = generate_questions(chunk_text)

            for generated_q in question_response.questions:
                question = generated_q.question.strip()
                category = generated_q.category

                question_key = question.lower()

                if question_key in seen_questions:
                    continue

                seen_questions.add(question_key)

                answer = generate_answer(
                    question=question,
                    category=category,
                    chunk_text=chunk_text
                )

                qa_pair = QAPair(
                    instruction="Answer the question using Uber's 2024 Annual Report.",
                    input=question,
                    output=answer,
                    source_chunk_id=chunk_id,
                    page_reference=page_reference,
                    category=category
                )

                qa_pairs.append(qa_pair.model_dump())

        except (ValidationError, json.JSONDecodeError) as e:
            print(f"[VALIDATION ERROR] chunk_id={chunk_id}: {e}")

        except Exception as e:
            print(f"[ERROR] chunk_id={chunk_id}: {e}")

    return qa_pairs


# ---------------------------------------------------------
# Train/Test Split
# ---------------------------------------------------------

def split_dataset(
    qa_pairs: List[dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple[list, list]:
    random.seed(seed)

    shuffled = qa_pairs.copy()
    random.shuffle(shuffled)

    split_index = int(len(shuffled) * train_ratio)

    train_data = shuffled[:split_index]
    test_data = shuffled[split_index:]

    return train_data, test_data


# ---------------------------------------------------------
# Dataset Summary
# ---------------------------------------------------------

def print_dataset_summary(qa_pairs: List[dict], train_data: List[dict], test_data: List[dict]) -> None:
    df = pd.DataFrame(qa_pairs)

    print("\n========== DATASET SUMMARY ==========")
    print(f"Total Q/A pairs: {len(qa_pairs)}")
    print(f"Train pairs: {len(train_data)}")
    print(f"Golden test pairs: {len(test_data)}")

    if not df.empty:
        print("\nCategory Distribution:")
        print(df["category"].value_counts())

        print("\nSource Chunk Count:")
        print(df["source_chunk_id"].nunique())

    print("=====================================\n")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)

    print(f"Loaded chunks: {len(chunks)}")

    qa_pairs = generate_qa_pairs(chunks)

    if not qa_pairs:
        raise ValueError("No Q/A pairs were generated. Check chunk quality or API setup.")

    train_data, test_data = split_dataset(qa_pairs)

    save_json(qa_pairs, QA_PAIRS_PATH)
    save_jsonl(train_data, TRAIN_PATH)
    save_jsonl(test_data, TEST_PATH)

    print_dataset_summary(qa_pairs, train_data, test_data)

    print(f"Saved qa_pairs.json to: {QA_PAIRS_PATH}")
    print(f"Saved train.jsonl to: {TRAIN_PATH}")
    print(f"Saved golden_test_set.jsonl to: {TEST_PATH}")


if __name__ == "__main__":
    main()