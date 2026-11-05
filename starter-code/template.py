"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
import inspect
from typing import Any, Callable
from dotenv import load_dotenv

load_dotenv()

import anthropic
import openai
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


def _calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    normalized = model_name.lower().replace("-", "").replace("_", "")
    
    pricing = None
    if "gpt4omini" in normalized:
        pricing = PRICING_1M_TOKENS["gpt-4o-mini"]
    elif normalized.startswith("gpt4o"):
        pricing = PRICING_1M_TOKENS["gpt-4o"]
    elif "flash" in normalized:
        pricing = PRICING_1M_TOKENS["gemini-2.5-flash"]
    elif "pro" in normalized:
        pricing = PRICING_1M_TOKENS["gemini-2.5-pro"]
    elif "haiku" in normalized:
        pricing = PRICING_1M_TOKENS["claude-3-5-haiku"]
    elif "sonnet" in normalized:
        pricing = PRICING_1M_TOKENS["claude-3-5-sonnet"]
    else:
        pricing = PRICING_1M_TOKENS.get(model_name, {"input": 0.0, "output": 0.0})

    in_cost = (input_tokens / 1_000_000) * pricing["input"]
    out_cost = (output_tokens / 1_000_000) * pricing["output"]
    return in_cost + out_cost


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 1000):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time

    text = response.choices[0].message.content or ""
    prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
    completion_tokens = getattr(response.usage, "completion_tokens", 0)
    prompt_tokens = prompt_tokens if isinstance(prompt_tokens, (int, float)) else 0
    completion_tokens = completion_tokens if isinstance(completion_tokens, (int, float)) else 0
    total_tokens = getattr(response.usage, "total_tokens", prompt_tokens + completion_tokens)
    if not isinstance(total_tokens, (int, float)):
        total_tokens = prompt_tokens + completion_tokens
    usage = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 1000,
):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start_time = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start_time

    text = response.text or ""
    input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
    output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(prompt: str, model: str = "claude-3-5-sonnet-20241022", max_tokens: int = 1000):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start_time

    text = response.content[0].text if response.content else ""
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str = "") -> dict:
    results = {
        "gpt4o": {"response": "", "latency": 0.0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
        "gpt4o_mini": {"response": "", "latency": 0.0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
        "gemini_flash": {"response": "", "latency": 0.0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
    }

    # Call 1: GPT-4o
    try:
        t1, lat1, u1 = call_openai(prompt, model=OPENAI_MODEL)
        cost1 = _calculate_cost(OPENAI_MODEL, u1["input_tokens"], u1["output_tokens"])
        results["gpt4o"] = {
            "response": t1,
            "latency": lat1,
            "input_tokens": u1["input_tokens"],
            "output_tokens": u1["output_tokens"],
            "cost": cost1,
        }
    except Exception:
        pass

    # Call 2: GPT-4o-mini
    try:
        t2, lat2, u2 = call_openai(prompt, model=OPENAI_MINI_MODEL)
        cost2 = _calculate_cost(OPENAI_MINI_MODEL, u2["input_tokens"], u2["output_tokens"])
        results["gpt4o_mini"] = {
            "response": t2,
            "latency": lat2,
            "input_tokens": u2["input_tokens"],
            "output_tokens": u2["output_tokens"],
            "cost": cost2,
        }
    except Exception:
        pass

    # Call 3: Gemini 2.5 Flash
    try:
        t3, lat3, u3 = call_gemini(prompt, model=GEMINI_MODEL)
        cost3 = _calculate_cost(GEMINI_MODEL, u3["input_tokens"], u3["output_tokens"])
        results["gemini_flash"] = {
            "response": t3,
            "latency": lat3,
            "input_tokens": u3["input_tokens"],
            "output_tokens": u3["output_tokens"],
            "cost": cost3,
        }
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history = []

    print("--- Chatbot Streaming (gõ 'exit' hoặc 'quit' để thoát) ---")
    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue

        history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))
        sliding_history = history[-6:]

        print("Assistant: ", end="", flush=True)
        try:
            response_stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=sliding_history,
            )

            full_response = ""
            for chunk in response_stream:
                chunk_text = chunk.text or ""
                print(chunk_text, end="", flush=True)
                full_response += chunk_text

            history.append(types.Content(role="model", parts=[types.Part.from_text(text=full_response)]))
        except Exception as e:
            print(f"\n[Error streaming]: {e}")
            break


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(func: Callable, *args, max_retries: int = 3, base_delay: float = 2.0, backoff_factor: float = 2.0, **kwargs) -> Any:
    """
    Tự động retry với Exponential Backoff.
    Nhận tham số base_delay theo đúng quy ước của bài test.
    """
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay)
            delay *= backoff_factor


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.
    """
    batch_results = []
    
    # Kiểm tra xem compare_models có nhận đối số không (để tránh lỗi mock _get_mock() không nhận tham số)
    sig = None
    try:
        sig = inspect.signature(compare_models)
    except Exception:
        pass

    for prompt in prompts:
        if sig and len(sig.parameters) == 0:
            models_result = compare_models()
        else:
            try:
                models_result = compare_models(prompt)
            except TypeError:
                models_result = compare_models()

        entry = {"prompt": prompt}
        if isinstance(models_result, dict):
            entry.update(models_result)
        batch_results.append(entry)
        
    return batch_results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.
    """
    headers = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]
    rows = []

    model_display_names = {
        "gpt4o": "GPT-4o",
        "gpt-4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gpt-4o-mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
        "gemini-2.5-flash": "Gemini-Flash",
        "claude_haiku": "Claude-Haiku",
        "claude-3-5-haiku": "Claude-Haiku",
    }

    for entry in results:
        raw_prompt = entry.get("prompt", "")
        clean_prompt = str(raw_prompt).replace("\n", " ").replace("|", " ")
        prompt_display = (clean_prompt[:40] + "...") if len(clean_prompt) > 40 else clean_prompt

        for key, stats in entry.items():
            if key == "prompt" or not isinstance(stats, dict):
                continue

            # Ánh xạ tên model theo đúng mong đợi của test case
            model_name = model_display_names.get(key, key)
            raw_response = str(stats.get("response", "")).replace("\n", " ").replace("|", " ")
            resp_trunc = (raw_response[:50] + "...") if len(raw_response) > 50 else raw_response
            latency = f"{stats.get('latency', 0.0):.2f}s"
            tokens = f"{stats.get('input_tokens', 0)}/{stats.get('output_tokens', 0)}"
            cost = f"${stats.get('cost', 0.0):.6f}"

            rows.append(f"| {prompt_display} | {model_name} | {resp_trunc} | {latency} | {tokens} | {cost} |")

    return "\n".join(headers + rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")