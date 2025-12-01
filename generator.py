# generator.py
# Local LLM generation using Hugging Face transformers (flan-t5-small)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

MODEL_NAME = "google/flan-t5-small"  # small and works on CPU (but slower)
device = 0 if torch.cuda.is_available() else -1

_tokenizer = None
_model = None
_pipeline = None

def load_generator():
    global _tokenizer, _model, _pipeline
    if _pipeline is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        if torch.cuda.is_available():
            _model = _model.to("cuda")
            _pipeline = pipeline("text2text-generation", model=_model, tokenizer=_tokenizer, device=0)
        else:
            _pipeline = pipeline("text2text-generation", model=_model, tokenizer=_tokenizer, device=-1)
    return _pipeline

def generate_answer(prompt, max_length=256, do_sample=False):
    pipe = load_generator()
    out = pipe(prompt, max_length=max_length, do_sample=do_sample, truncation=True)
    if isinstance(out, list) and len(out) > 0:
        return out[0]["generated_text"]
    return ""
