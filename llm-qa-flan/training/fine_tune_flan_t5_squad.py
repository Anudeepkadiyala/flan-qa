import argparse, os, numpy as np
from datasets import load_dataset
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer, TrainingArguments,              # keep if you want, but we’ll use the seq2seq ones
    Seq2SeqTrainer, Seq2SeqTrainingArguments # <-- add these
)


def preprocess_function(examples, tokenizer, max_input_length=384, max_target_length=64):
    inputs, targets = [], []
    for context, question, answers in zip(examples["context"], examples["question"], examples["answers"]):
        inputs.append(f"question: {question}  context: {context}")
        targets.append(answers["text"][0] if answers and answers["text"] else "")
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="google/flan-t5-small")
    p.add_argument("--output_dir", type=str, default="outputs/flan-t5-small-squad")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    # NEW: quick subset flags
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    return p.parse_args()

def main():
    args = build_args()
    print("Transformers:", transformers.__version__)

    # Robust dataset load. If cache is corrupted, clear SQuAD cache and retry.
    try:
        dataset = load_dataset("squad")
    except Exception as e:
        print("Load failed once; trying after clearing local SQuAD cache…", e)
        # best-effort cache clean
        hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
        for name in ("squad",):
            pth = os.path.join(hf_cache, name)
            if os.path.exists(pth):
                try:
                    import shutil
                    shutil.rmtree(pth)
                    print("Removed cache:", pth)
                except Exception as _:
                    pass
        dataset = load_dataset("squad")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Optional small subsets for quick demo runs
    train_ds = dataset["train"]
    eval_ds = dataset["validation"]
    if args.max_train_samples:
        n = min(args.max_train_samples, len(train_ds))
        train_ds = train_ds.select(range(n))
    if args.max_eval_samples:
        n = min(args.max_eval_samples, len(eval_ds))
        eval_ds = eval_ds.select(range(n))

    # Tokenize
    remove_cols = list(set(train_ds.column_names + eval_ds.column_names))
    tokenized_train = train_ds.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True, remove_columns=remove_cols, desc="Tokenizing train"
    )
    tokenized_eval = eval_ds.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True, remove_columns=remove_cols, desc="Tokenizing eval"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Version-safe TrainingArguments: try evaluation_strategy, fall back to eval_strategy
    try:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            seed=args.seed,
            logging_steps=50,
            predict_with_generate=True,
            bf16=False, fp16=False,
            report_to=[],  # safer across versions
        )
    except TypeError:
        training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        seed=args.seed,
        logging_steps=50,
        predict_with_generate=True,   # now valid
        bf16=False,
        fp16=False,
        report_to=[],                 # quieter across versions
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Training…")
    trainer.train()
    print("Saving model to:", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
