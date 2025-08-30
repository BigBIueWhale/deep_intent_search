import os
from dotenv import load_dotenv
import ollama
from core.tokens import count_tokens

# Load env once here; callers can also call load_dotenv earlier safely.
load_dotenv()

# --- Configuration ---

def _normalize_host(raw: str | None) -> str:
    """
    Accepts IP+Port as a single string variable (env). Defaults to 127.0.0.1:11434.
    Adds http:// if missing. If only IP is provided (no ':'), appends :11434.
    """
    default = "127.0.0.1:11434"
    host = (raw or default).strip()
    if ":" not in host:
        host = f"{host}:11434"
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
    return host

def get_client() -> ollama.Client | None:
    """
    Singleton-style accessor for the Ollama client. No raw HTTP is used.
    """
    host = _normalize_host(os.environ.get("OLLAMA_HOST"))
    try:
        return ollama.Client(host=host)
    except Exception as e:
        print(f"Error initializing Ollama client at {host}: {e}")
        return None

def get_model_name() -> str:
    """
    Centralized model name for easy switching later.
    Defaults to qwen3:32b as requested.
    """
    return os.environ.get("OLLAMA_MODEL", "qwen3:32b")

# Advanced parameters for qwen3:32b (applied on every request)
_QWEN3_32B_OPTIONS = {
    "num_ctx": 24000,       # Context Length
    # Setting -1 (infinite) would cause infinite generation once in a while.
    # Infinite generations are observed to be exactly 239,998 thinking tokens
    # plus 2 response tokens.
    # To avoid the issue of Ollama call getting stuck waiting for almost 2 hours,
    # grinding the GPU for nothing, we'll set num_predict to a value greater then
    # the longest successful response I've observed so far (6659 tokens).
    "num_predict": 8192,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "num_gpu": 65,          # Layers to offload
}

def get_ollama_options() -> dict:
    """
    Returns the per-request options dict for Ollama. Centralized so we can
    tweak or swap models later without touching the scripts.
    """
    return dict(_QWEN3_32B_OPTIONS)

# For debug statistics
def print_stats(response: ollama.ChatResponse) -> str | None:
    if None in [response.prompt_eval_duration, response.prompt_eval_count,
                response.eval_duration, response.eval_count]:
        return None
    prefill_speed = response.prompt_eval_count / (response.prompt_eval_duration / 1e9)
    generation_speed = response.eval_count / (response.eval_duration / 1e9)
    thinking_text = response.message.thinking
    thinking_tokens = count_tokens(thinking_text) if thinking_text else 0
    return f"prefill_speed: {prefill_speed:.2f}(tok/sec), generation_speed: {generation_speed:.2f}(tok/sec)" + \
          f"\nprompt: {response.prompt_eval_count}(tok), think: {thinking_tokens}(tok) + response: {response.eval_count - thinking_tokens}(tok)"
