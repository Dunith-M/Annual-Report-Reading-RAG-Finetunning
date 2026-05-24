import os
import yaml
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from peft import LoraConfig, prepare_model_for_kbit_training

# TRL 1.4 reads bundled chat-template files without specifying an encoding.
# On Windows, Python may default to cp1252 and fail on UTF-8 template files.
_path_read_text = Path.read_text


def _read_text_utf8_by_default(self, encoding=None, errors=None, newline=None):
    return _path_read_text(
        self,
        encoding=encoding or "utf-8",
        errors=errors,
        newline=newline
    )


Path.read_text = _read_text_utf8_by_default

from trl import SFTTrainer, SFTConfig

from src.finetuning.dataset_builder import build_sft_dataset


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_compute_dtype() -> torch.dtype:
    """
    T4 usually supports fp16 safely.
    A100/L4/H100 can use bf16, but fp16 is safer for Colab T4.
    """
    return torch.float16


def plot_training_loss(trainer, output_path: str) -> None:
    logs = trainer.state.log_history

    steps = []
    losses = []

    for item in logs:
        if "loss" in item and "step" in item:
            steps.append(item["step"])
            losses.append(item["loss"])

    if not losses:
        print("No loss values found in trainer logs.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker="o")
    plt.xlabel("Training Step")
    plt.ylabel("Training Loss")
    plt.title("Fine-Tuning Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved loss curve to: {output_path}")


def train_intern(config_path: str = "configs/model_config.yaml") -> None:
    config = load_yaml(config_path)["fine_tuning"]

    base_model_id = config["base_model_id"]
    train_file = config["train_file"]
    output_dir = config["output_dir"]
    logging_dir = config["logging_dir"]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs("artifacts/reports", exist_ok=True)

    dataset = build_sft_dataset(train_file)

    compute_dtype = get_compute_dtype()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        logging_dir=logging_dir,

        max_length=config["max_seq_length"],
        dataset_text_field="text",

        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"],

        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],

        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],

        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],

        fp16=True,
        bf16=False,

        optim="paged_adamw_8bit",
        report_to=["tensorboard"],

        save_total_limit=2,
        packing=False
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Saving LoRA adapter...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    loss_curve_path = "artifacts/reports/training_loss_curve.png"
    plot_training_loss(trainer, loss_curve_path)

    print("Fine-tuning completed.")
    print(f"LoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    train_intern()
