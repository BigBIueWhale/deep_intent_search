# /core/llm.py

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from core.tokens import count_tokens

# Load env once here; callers can also call load_dotenv earlier safely.
load_dotenv()


# --- Configuration & helpers ---

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
    return host.rstrip("/")


def _base_url() -> str:
    return _normalize_host(os.environ.get("OLLAMA_HOST"))


def _request_timeout_seconds() -> float:
    # Hard ceiling so calls cannot hang forever.
    # You can override in .env; defaults are generous for large models.
    return float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "300"))  # 5 minutes


# --- Public API compatibility layer ---

@dataclass
class _Message:
    role: str = "assistant"
    content: str = ""
    thinking: Optional[str] = None  # extracted if available (or from <think>...</think>)


@dataclass
class ChatResponse:
    """
    Drop-in stand-in for `ollama.ChatResponse` attributes your code uses.
    """
    message: _Message
    # The following names match your print_stats() expectations
    prompt_eval_duration: Optional[int] = None   # nanoseconds
    prompt_eval_count: Optional[int] = None
    eval_duration: Optional[int] = None          # nanoseconds (includes think+response)
    eval_count: Optional[int] = None             # tokens generated (incl. think where applicable)
    ran_out_of_tokens: bool = False


def get_client() -> httpx.Client | None:
    """
    Returns a configured httpx.Client. Keeping the name for backward compatibility.
    """
    # httpx separates connect/read/write; we use a firm read timeout as our "total" cap
    timeout_total = _request_timeout_seconds()
    try:
        client = httpx.Client(
            timeout=httpx.Timeout(
                connect=10.0,    # fail fast if daemon is down
                read=timeout_total,  # cap total wait on non-stream responses
                write=10.0,
                pool=None,       # default pool is fine; leaving explicit for clarity
            )
        )
        return client
    except Exception as e:
        raise RuntimeError(f"Error initializing httpx client for Ollama at {_base_url()}: {e}")
    
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

# Advanced parameters for qwen3:32b, applied on every request.
# Shared across think & no-think.
_QWEN3_32B_BASE_OPTIONS = {
    # Good context length value for 32GB VRAM and flash attention enabled
    # Ends up using ~31 GB in "ollama ps" when context length is full.
    "num_ctx": 19456,  # 19k

    # Setting -1 (infinite) would cause infinite generation once in a while.
    # Infinite generations are observed to be exactly 239,998 thinking tokens
    # plus 2 response tokens.
    # Avoid the issue of Ollama call getting stuck waiting for almost 2 hours,
    # grinding the GPU for nothing on gibberish.
    "num_predict": 11264,

    # Shared sampling/guardrails for both modes
    "top_k": 20,
    "min_p": 0.0,
    "repeat_penalty": 1.0,

    # Layers to offload, all of them.
    "num_gpu": 65,
}

# Think mode settings recommended by Alibaba
_QWEN3_32B_THINK_OPTIONS = {
    **_QWEN3_32B_BASE_OPTIONS,
    "temperature": 0.6,
    "top_p": 0.95,
}

# No-think mode settings recommended by Alibaba
_QWEN3_32B_NO_THINK_OPTIONS = {
    **_QWEN3_32B_BASE_OPTIONS,
    "temperature": 0.7,
    "top_p": 0.8,
}

# Advanced parameters for qwen3:30b-a3b-thinking-2507-q4_K_M (applied on every request).
# Not the default, because it's much worse at instruction following.
# For example, the 30B_A3B version will often justify relevance based on content
# outside of the section of interest.
_QWEN3_30B_A3B_THINKING_OPTIONS = {
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

# Advanced parameters for qwen3:30b-a3b-instruct-2507-q4_K_M (applied on every request).
# Matches official Qwen/Unsloth guidance for Instruct 2507, but keeps the same context length
# and layer offload as the thinking variant by design.
_QWEN3_30B_A3B_INSTRUCT_OPTIONS = {
    # Good context length value for 32GB VRAM and flash attention enabled
    # Keep identical to the thinking variant per requirements.
    "num_ctx": 57344, # 56k
    # Instruct variants don't "think" as long; you can still override at call-site if needed.
    # Using the "magic number" 16,384 instead of the 11,264 previously written here —
    # because the official page for Qwen3-30B-A3B-Instruct-2507 says:
    #   "Adequate Output Length: We recommend using an output length of 16,384 tokens for most queries,
    #    which is adequate for instruct models."
    # Source: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
    "num_predict": 16384,
    # Official guidance for Instruct-2507:
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.8,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    # Layers to offload; keep identical to thinking variant.
    "num_gpu": 49,
}

_GLM_4_32B_OPTIONS = {
    # GLM4 has crazy efficient context window.
    # It has 48 q heads and only 2 kv heads, a 24:1 ratio which is pretty high.
    # We turn on full context capability of the model
    "num_ctx": 32768,
    # Just a number to avoid infinite generation
    "num_predict": 8192,
}

_GEMMA3_27B_OPTIONS = {
    # Gemma3 is unique in the fact it was pretrained on
    # 32k token context length, unlike qwen3 which was mostly
    # trained on 4096 context length.
    "num_ctx": 32768,
    # Any more than this is undefined behaviour according to Google
    # The model was never trained on outputting more than 8192 tokens.
    "num_predict": 8192,
}

def get_ollama_options(model: str, please_no_thinking: bool) -> dict:
    if model == "qwen3:32b":
        return dict(
            _QWEN3_32B_NO_THINK_OPTIONS
            if please_no_thinking
            else _QWEN3_32B_THINK_OPTIONS
        )
    if model == "qwen3:30b-a3b-thinking-2507-q4_K_M":
        return dict(_QWEN3_30B_A3B_THINKING_OPTIONS)
    if model == "qwen3:30b-a3b-instruct-2507-q4_K_M":
        return dict(_QWEN3_30B_A3B_INSTRUCT_OPTIONS)
    if model == "JollyLlama/GLM-4-32B-0414-Q4_K_M":
        return dict(_GLM_4_32B_OPTIONS)
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

def _supports_qwen3_hybrid(model: str) -> bool:
    return model in {
        "qwen3:32b",
    }

_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)


def _extract_thinking(message_obj: dict, can_think: bool, content: str) -> Optional[str]:
    """
    Extract a 'thinking' trace if present. Ollama sometimes emits it as a separate
    field, or inline inside <think>...</think> tags. We support both.
    """
    # 1) explicit field (future / variant-friendly)
    if isinstance(message_obj, dict) and "thinking" in message_obj:
        if isinstance(message_obj["thinking"], str):
            return message_obj["thinking"]

    # 2) inline tags
    if can_think and content:
        m = _THINK_TAG_RE.search(content)
        if m:
            return m.group(1).strip()

    return None


def chat_complete(
    messages: list[dict[str, str]],
    role: str,
    client: httpx.Client,
    max_completion_tokens: int, # Only used in non-thinking mode
    please_no_thinking: bool,   # Only used for hybrid reasoning / non reasoning models
    require_json: bool = True,  # Only used for models that don't emit <think> token
) -> ChatResponse:
    """
    Non-streaming call to Ollama's /api/chat using httpx, with hard read timeout.
    Not using Ollama's official Python client because it doesn't provide a timeout option.
    Single entrypoint so callers do not duplicate flags:
    - thinking is enabled for thinking-capable models only
    - when not thinking and JSON is desired, we set format="json"
    """
    if client is None:
        raise RuntimeError("httpx client is not initialized")

    model = get_model_name(role)
    options = get_ollama_options(model, please_no_thinking)
    can_think = _supports_thinking(model)
    is_hybrid = _supports_qwen3_hybrid(model)

    hybrid_nothink_switch = is_hybrid and please_no_thinking

    if hybrid_nothink_switch:
        # Insert "/no_think" at the beginning of the system prompt
        system_prompt = messages[0]
        system_prompt["content"] = f"/no_think {system_prompt['content']}"

    if hybrid_nothink_switch or not can_think:
        options["num_predict"] = max_completion_tokens

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "options": options,
        "stream": False,
    }

    # Strict JSON output enforced by Ollama doesn't work together with "<think>" tags.
    if require_json and not can_think:
        payload["format"] = "json"

    if can_think and not hybrid_nothink_switch:
        # Simulate Ollama response structure
        if can_think:
            payload["think"] = True

    url = f"{_base_url()}/api/chat"

    # Make the request; httpx read-timeout caps total wait for the non-streaming body.
    # If Ollama wedges without sending bytes, you get a TimeoutException instead of a forever hang.
    resp = client.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()

    # Build a compatibility response
    msg = data.get("message") or {}
    content = msg.get("content") or ""

    thinking_text = _extract_thinking(msg, can_think, content)

    # Stats: use Ollama JSON keys if present; set None otherwise
    # Units from Ollama are nanoseconds for *_duration fields.
    prompt_eval_duration = data.get("prompt_eval_duration")
    prompt_eval_count = data.get("prompt_eval_count")
    eval_duration = data.get("eval_duration")
    eval_count = data.get("eval_count")
    done_reason = data.get("done_reason", "")

    # Simulate Ollama response structure
    return ChatResponse(
        message=_Message(
            role=msg.get("role", "assistant"),
            content=content,
            thinking=thinking_text,
        ),
        prompt_eval_duration=prompt_eval_duration,
        prompt_eval_count=prompt_eval_count,
        eval_duration=eval_duration,
        eval_count=eval_count,
        ran_out_of_tokens=(done_reason.lower() == 'length')
    )


# For debug statistics
def print_stats(response: ChatResponse) -> str | None:
    if None in [response.prompt_eval_duration, response.prompt_eval_count,
                response.eval_duration, response.eval_count]:
        return None
    try:
        prefill_speed = response.prompt_eval_count / (response.prompt_eval_duration / 1e9)
        generation_speed = response.eval_count / (response.eval_duration / 1e9)
    except ZeroDivisionError:
        prefill_speed = 0.0
        generation_speed = 0.0

    thinking_text = response.message.thinking or ""
    thinking_tokens = count_tokens(thinking_text) if thinking_text else 0
    return (
        f"prefill_speed: {prefill_speed:.2f}(tok/sec), "
        f"generation_speed: {generation_speed:.2f}(tok/sec)"
        + f"\nprompt: {response.prompt_eval_count}(tok), "
        f"think: {thinking_tokens}(tok) + response: {response.eval_count - thinking_tokens}(tok)"
    )
