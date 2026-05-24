import json
from pathlib import Path
from typing import Dict, Any

from datasets import Dataset


SYSTEM_PROMPT = """
You are The Intern, a fine-tuned assistant trained on Uber's 2024 Annual Report.
Answer in a clear, professional, investor-report style.
If the question asks for exact financial numbers and you are unsure, say that the number should be verified against the source report.
Do not invent figures.
""".strip()


def load_jsonl(path: str) -> list[Dict[str, Any]]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def build_prompt(example: Dict[str, Any]) -> str:
    instruction = example.get("instruction", "").strip()
    user_input = example.get("input", "").strip()
    output = example.get("output", "").strip()

    user_message = f"{instruction}\n\nQuestion:\n{user_input}"

    text = f"""<s>[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

{user_message} [/INST]
{output}</s>"""

    return text


def build_sft_dataset(train_file: str) -> Dataset:
    records = load_jsonl(train_file)

    formatted_records = []
    for row in records:
        formatted_records.append({
            "text": build_prompt(row),
            "source_chunk_id": row.get("source_chunk_id", ""),
            "page_reference": row.get("page_reference", ""),
            "category": row.get("category", "")
        })

    return Dataset.from_list(formatted_records)


if __name__ == "__main__":
    dataset = build_sft_dataset("data/processed/train.jsonl")
    print(dataset)
    print(dataset[0]["text"][:1000])