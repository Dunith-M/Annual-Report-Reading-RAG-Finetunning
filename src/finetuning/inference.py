import yaml
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from peft import PeftModel


SYSTEM_PROMPT = """
You are The Intern, a fine-tuned assistant trained on Uber's 2024 Annual Report.
Answer in a clear, professional, investor-report style.
If the question asks for exact financial numbers and you are unsure, say that the number should be verified against the source report.
Do not invent figures.
""".strip()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_inference_prompt(question: str) -> str:
    return f"""<s>[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

Answer the question using Uber's 2024 Annual Report.

Question:
{question} [/INST]"""


def load_intern_model(config_path: str = "configs/model_config.yaml"):
    config = load_yaml(config_path)["fine_tuning"]

    base_model_id = config["base_model_id"]
    adapter_dir = config["output_dir"]

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        trust_remote_code=True,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    return model, tokenizer


def query_intern(question: str) -> str:
    """
    Loads base model + LoRA adapter and answers the question.
    """
    model, tokenizer = load_intern_model()

    prompt = build_inference_prompt(question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]", 1)[-1].strip()

    return decoded


if __name__ == "__main__":
    answer = query_intern("What are Uber's main business segments?")
    print(answer)