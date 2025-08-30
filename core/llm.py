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

def get_model_name(role: str | None = None) -> str:
    """
    Role must be 'judge' or 'splitter'. No fallback. Force explicit config.
    """
    if role not in {"judge", "splitter"}:
        raise ValueError("get_model_name(role): role must be 'judge' or 'splitter'.")

    env_key = "OLLAMA_MODEL_JUDGE" if role == "judge" else "OLLAMA_MODEL_SPLITTER"
    name = os.environ.get(env_key, "").strip()
    if not name:
        raise ValueError(f"Missing required env var {env_key}. See README for .env options")
    
    return name

# Advanced parameters for qwen3:32b (applied on every request)
_QWEN3_32B_OPTIONS = {
    # Good context length value for 32GB VRAM and flash attention enabled
    # Ends up using ~31 GB in "ollama ps" when context length is full.
    "num_ctx": 19456, # 19k
    # Setting -1 (infinite) would cause infinite generation once in a while.
    # Infinite generations are observed to be exactly 239,998 thinking tokens
    # plus 2 response tokens.
    # Avoid the issue of Ollama call getting stuck waiting for almost 2 hours,
    # grinding the GPU for nothing on gibberish.
    "num_predict": 11264,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "num_gpu": 65, # Layers to offload, all of them.
}

# Advanced parameters for qwen3:30b-a3b-thinking-2507-q4_K_M (applied on every request).
# Not the default, because it's much worse at instruction following.
# For example, the 30B_A3B version will often justify relevance based on content
# outside of the section of interest.
_QWEN3_30B_A3B_OPTIONS = {
    # Good context length value for 32GB VRAM and flash attention enabled
    # Ends up using ~31 GB in "ollama ps" when context length is full.
    "num_ctx": 57344, # 56k
    "num_predict": 49152, # 30b-a3b tends to think a lot
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "num_gpu": 49, # Layers to offload, all of them.
}

_GEMMA3_27B_OPTIONS = {
    # Good context length value for 32GB VRAM and flash attention enabled
    # Ends up using ~32 GB in "ollama ps" when context length is full.
    "num_ctx": 81920,
    # Any more than this is undefined behaviour according to Google
    # The model was never trained on outputting more than 8192 tokens.
    "num_predict": 8192,
}

def get_ollama_options(model: str) -> dict:
    if model == "qwen3:32b":
        return dict(_QWEN3_32B_OPTIONS)
    if model == "qwen3:30b-a3b-thinking-2507-q4_K_M":
        return dict(_QWEN3_30B_A3B_OPTIONS)
    if model == "gemma3:27b":
        return dict(_GEMMA3_27B_OPTIONS)
    raise ValueError(
        f"Unrecognized OLLAMA_MODEL '{model}'. See README for .env options"
    )

def _supports_thinking(model: str) -> bool:
    return model in {
        "qwen3:32b",
        "qwen3:30b-a3b-thinking-2507-q4_K_M",
    }

def chat_complete(
    messages: list[dict],
    role: str,
    client: ollama.Client,
    require_json: bool = True,
) -> ollama.ChatResponse:
    """
    Single entrypoint so callers do not duplicate flags:
    - thinking is enabled for thinking-capable models only
    - when not thinking and JSON is desired, we set format="json"
    """
    model = get_model_name(role)
    options = get_ollama_options(model)
    can_think = _supports_thinking(model)

    kwargs = {
        "model": model,
        "messages": messages,
        "options": options,
        "stream": False,
        "think": can_think,
    }
    # Strict JSON output enforced by Ollama doesn't work
    # together with "<think>" tags.
    if require_json and not can_think:
        kwargs["format"] = "json"

    return client.chat(**kwargs)

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
