import threading

# Centralized token counting using the Qwen3-32B tokenizer for better alignment.
# Falls back gracefully if transformers is unavailable.
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

_tokenizer = None
_tokenizer_lock = threading.Lock()

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    with _tokenizer_lock:
        if _tokenizer is None:
            if AutoTokenizer is not None:
                try:
                    _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
                except Exception:
                    _tokenizer = None
        return _tokenizer

def count_tokens(text: str) -> int:
    tok = _get_tokenizer()
    return len(tok.encode(text))
