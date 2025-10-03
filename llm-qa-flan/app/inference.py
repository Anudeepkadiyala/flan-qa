import os
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = os.getenv("MODEL_DIR", "outputs/flan-t5-small-squad")

class QAEngine:
    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None):
        self.model_dir = model_dir or MODEL_DIR
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
        if device:
            self.model.to(device)

    def answer(self, context: str, question: str, max_new_tokens: int = 64) -> str:
        prompt = f"question: {question}  context: {context}"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.model.device.type != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
