## Langchain

#### General points

- Parallel RAG Data Retrieval -> from langchain_core.runnables import RunnableParallel, RunnablePassthrough -> This pattern is used to run independent tasks at the same time. In a RAG application, you often want to fetch relevant documents from a vector store and preserve the user's original question simultaneously before passing both to the prompt.
  
- LangChain Expression Language (LCEL) is a declarative way to compose and chain together modular AI components like prompts, models, and output parsers using the native pipe operator (|)

#### Text Splitters 
- LangChain offers several splitters for different content types. In your LangChain 0.3.x setup, the dedicated package is already listed in requirements, so prefer imports from langchain_text_splitters:

CharacterTextSplitter: splits on a chosen separator, such as paragraphs or newlines. Simple, but may produce chunks that exceed the size limit.
TokenTextSplitter: splits by model tokens, useful when you need to stay within an LLM’s token budget.
MarkdownHeaderTextSplitter: splits Markdown by headings and can preserve heading metadata.
HTMLHeaderTextSplitter: splits HTML by heading structure.
RecursiveJsonSplitter: splits nested JSON while trying to preserve its structure.
RecursiveCharacterTextSplitter.from_language(...): adapts recursive separators for languages such as Python, JavaScript, or Markdown.


## Langgraph



## LLM Management

- Exponential backoff & retries ->
- we manipulate wait times here using the formula 


| **Backoff Type** | **Core Formula (No Jitter)** | **Formula with Full Jitter** |
|---|---|---|
| **Exponential** | `min(Base * (B ^ attempt), Max)` | `Random(0, min(Base * (B ^ attempt), Max))` |
| **Linear** | `min(Base + (Step * attempt), Max)` | `Random(0, min(Base + (Step * attempt), Max))` |
| **Constant / Fixed** | `Base` | `Base + Random(-Jitter, Jitter)` |
| **Polynomial / Quadratic** | `min(Base * (attempt ^ 2), Max)` | `Random(0, min(Base * (attempt ^ 2), Max))` |
| **Fibonacci** | `min(Base * Fib(attempt), Max)` | `Random(0, min(Base * Fib(attempt), Max))` |
| **Decorrelated Jitter** | *N/A (Inherent jitter)* | `min(Max, Random(Base, Previous_Delay * 3))` |



  in python it can be implemented like -
```
"""Azure OpenAI client wrapper with retry/backoff and lightweight call instrumentation.

The notebook owns secret retrieval (``dbutils``); it passes a token-getter callable in here so
this module stays runtime-agnostic and unit-testable.
"""
import logging
import threading
import time
from typing import Any, Callable, Optional

import requests

# Models that do NOT support the temperature parameter.
NO_TEMPERATURE_MODELS = ("o3-mini", "o3", "gpt-5-mini", "gpt-5", "gpt-5.4")


def make_bearer_token_getter(client_id: str, client_secret: str, token_endpoint: str) -> Callable[[], str]:
    """Return a zero-arg callable that fetches a fresh Azure OpenAI bearer token."""
    def _get_token() -> str:
        resp = requests.post(token_endpoint, data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "openid",
        })
        if resp.status_code == 200:
            return resp.json().get("access_token")
        raise RuntimeError(f"Failed to get bearer token: {resp.status_code} - {resp.text}")
    return _get_token


class LLMClient:
    """Thread-safe Azure OpenAI wrapper. Re-auths on 401, backs off on rate limits, and
    tracks call/latency/retry/failure counters for per-stage instrumentation."""

    def __init__(self, endpoint: str, token_getter: Callable[[], str], api_version: str = "2025-01-01-preview",
                 max_retries: int = 10, retry_delay: int = 60):
        self._endpoint = endpoint
        self._token_getter = token_getter
        self._api_version = api_version
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._client = self._new_client()
        self.stats = {"calls": 0, "failures": 0, "retries": 0, "total_latency_s": 0.0}

    def _new_client(self) -> Any:
        # Imported lazily so the package imports without ``openai`` installed (it is
        # ``%pip install``-ed in the notebook; local unit tests don't need it).
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_version=self._api_version,
            azure_endpoint=self._endpoint,
            azure_ad_token=self._token_getter(),
        )

    def reset_stats(self) -> None:
        with self._stats_lock:
            self.stats.update(calls=0, failures=0, retries=0, total_latency_s=0.0)

    def _record(self, latency_s: float, retries: int, failed: bool) -> None:
        with self._stats_lock:
            self.stats["calls"] += 1
            self.stats["retries"] += retries
            self.stats["total_latency_s"] += latency_s
            if failed:
                self.stats["failures"] += 1

    def get_response(self, system_prompt: str, user_prompt: str, model: str,
                     max_retries: Optional[int] = None, retry_delay: Optional[int] = None) -> Optional[str]:
        """Call the chat completions API with JSON response format and resilient retries.
        ``max_retries``/``retry_delay`` default to the values set on the client."""
        max_retries = self._max_retries if max_retries is None else max_retries
        retry_delay = self._retry_delay if retry_delay is None else retry_delay
        t0 = time.time()
        retries_used = 0

        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                if model not in NO_TEMPERATURE_MODELS:
                    kwargs["temperature"] = 0.01

                response = self._client.chat.completions.create(**kwargs)
                self._record(time.time() - t0, retries_used, failed=False)
                return response.choices[0].message.content

            except Exception as e:
                retries_used += 1
                error_msg = str(e)
                logging.warning(f"Attempt {attempt+1}/{max_retries} failed: {error_msg}")

                if "401" in error_msg or "unauthorized" in error_msg.lower():
                    logging.warning("Re-authenticating with fresh bearer token...")
                    with self._client_lock:
                        self._client = self._new_client()
                    continue

                elif "Rate limit" in error_msg or "429" in error_msg or "timeout" in error_msg.lower():
                    # Honor server Retry-After; otherwise exponential backoff capped at retry_delay.
                    wait_time = min(retry_delay, 5 * (2 ** attempt))
                    if hasattr(e, "response") and hasattr(e.response, "headers"):
                        retry_after = e.response.headers.get("Retry-After")
                        if retry_after and str(retry_after).isdigit():
                            wait_time = int(retry_after) + 2
                    logging.info(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)

                elif "content_filter" in error_msg or "ResponsibleAIPolicyViolation" in error_msg:
                    logging.warning("Content filter triggered — skipping.")
                    self._record(time.time() - t0, retries_used, failed=True)
                    return None

                else:
                    if attempt == max_retries - 1:
                        logging.warning("Max retries reached — skipping.")
                        self._record(time.time() - t0, retries_used, failed=True)
                        return None
                    time.sleep(min(retry_delay, 5 * (2 ** attempt)))

        self._record(time.time() - t0, retries_used, failed=True)
        return None


class stage_timer:
    """Context manager that prints wall-clock + LLM-call deltas for a named stage."""

    def __init__(self, llm: LLMClient, name: str):
        self.llm = llm
        self.name = name

    def __enter__(self):
        self._t0 = time.time()
        self._c0 = self.llm.stats["calls"]
        return self

    def __exit__(self, *exc):
        dt = time.time() - self._t0
        dc = self.llm.stats["calls"] - self._c0
        print(f"    [timing] {self.name}: {dt:.1f}s across {dc} LLM calls")
        return False

```
