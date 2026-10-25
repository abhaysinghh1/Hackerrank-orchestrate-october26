"""
LLM Client wrapper with token/cost tracking and response caching.

Wraps Gemini API calls (text + multimodal vision) and tracks:
  - Provider and model name
  - Call count
  - Input tokens, output tokens
  - Estimated USD cost
  - In-memory response cache keyed by (model, prompt_hash, file_hash)
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Optional dependency: google-genai
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Optional dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------- Cost tables (approximate per-million-token prices) ---------- #
# Update these as pricing changes.
COST_PER_MILLION_TOKENS = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro":   {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro":   {"input": 1.25, "output": 5.00},
}

DEFAULT_COST = {"input": 0.15, "output": 0.60}  # Fallback


@dataclass
class CallRecord:
    """Tracks a single API call."""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool = False


@dataclass
class UsageTracker:
    """Aggregates token/cost stats across all calls."""
    calls: List[CallRecord] = field(default_factory=list)
    _cache: Dict[str, str] = field(default_factory=dict)

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def cached_calls(self) -> int:
        return sum(1 for c in self.calls if c.cached)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def by_model(self) -> Dict[str, Dict[str, Any]]:
        """Return per-model aggregates."""
        result: Dict[str, Dict[str, Any]] = {}
        for c in self.calls:
            key = f"{c.provider}/{c.model}"
            if key not in result:
                result[key] = {
                    "provider": c.provider, "model": c.model,
                    "calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "cost_usd": 0.0, "cached": 0,
                }
            result[key]["calls"] += 1
            result[key]["input_tokens"] += c.input_tokens
            result[key]["output_tokens"] += c.output_tokens
            result[key]["cost_usd"] += c.cost_usd
            if c.cached:
                result[key]["cached"] += 1
        return result

    def summary_text(self, total_cases: int = 0) -> str:
        """Render usage report as markdown text."""
        lines = ["# Token Usage and Cost Report", ""]
        by_model = self.by_model()
        for key, m in by_model.items():
            lines.append(f"## {key}")
            lines.append(f"- Model calls: {m['calls']} ({m['cached']} cache hits)")
            lines.append(f"- Input tokens: {m['input_tokens']:,}")
            lines.append(f"- Output tokens: {m['output_tokens']:,}")
            lines.append(f"- Total tokens: {m['input_tokens'] + m['output_tokens']:,}")
            lines.append(f"- Estimated cost: ${m['cost_usd']:.4f}")
            lines.append("")

        lines.append("## Overall Totals")
        lines.append(f"- Total model calls: {self.total_calls} ({self.cached_calls} cache hits)")
        lines.append(f"- Total input tokens: {self.total_input_tokens:,}")
        lines.append(f"- Total output tokens: {self.total_output_tokens:,}")
        lines.append(f"- Total tokens: {self.total_tokens:,}")
        lines.append(f"- Estimated total cost: ${self.total_cost:.4f}")
        if total_cases > 0:
            lines.append(f"- Average tokens per request: {self.total_tokens / total_cases:,.0f}")
            lines.append(f"- Estimated per-request cost: ${self.total_cost / total_cases:.6f}")
        return "\n".join(lines)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    costs = COST_PER_MILLION_TOKENS.get(model, DEFAULT_COST)
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000


def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
    return h.hexdigest()


class LLMClient:
    """Wrapper for Gemini text and vision calls with caching and tracking."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        provider: str = "google",
        api_key: Optional[str] = None,
        tracker: Optional[UsageTracker] = None,
    ):
        self.model = model
        self.provider = provider
        self.tracker = tracker or UsageTracker()

        if not HAS_GENAI:
            raise ImportError("google-genai package is required. Run: pip install google-genai")

        api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment or .env file.")

        self.client = genai.Client(api_key=api_key)

    def text(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        cache_key: Optional[str] = None,
    ) -> str:
        """Send a text prompt and return the model response text."""
        ck = cache_key or _hash_key(self.model, system_instruction, prompt)
        if ck in self.tracker._cache:
            # Record as a cached call with 0 tokens
            self.tracker.calls.append(CallRecord(
                provider=self.provider, model=self.model,
                input_tokens=0, output_tokens=0, cost_usd=0.0, cached=True
            ))
            return self.tracker._cache[ck]

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        text_out = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        cost = _estimate_cost(self.model, in_tok, out_tok)

        self.tracker.calls.append(CallRecord(
            provider=self.provider, model=self.model,
            input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=cost, cached=False
        ))
        self.tracker._cache[ck] = text_out
        return text_out

    def vision(
        self,
        prompt: str,
        image_path: str,
        system_instruction: str = "",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        cache_key: Optional[str] = None,
    ) -> str:
        """Send image + text prompt for multimodal analysis."""
        import pathlib
        img_bytes = pathlib.Path(image_path).read_bytes()
        ext = pathlib.Path(image_path).suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
        mime = mime_map.get(ext, "image/png")

        file_hash = hashlib.sha256(img_bytes).hexdigest()
        ck = cache_key or _hash_key(self.model, system_instruction, prompt, file_hash)
        if ck in self.tracker._cache:
            self.tracker.calls.append(CallRecord(
                provider=self.provider, model=self.model,
                input_tokens=0, output_tokens=0, cost_usd=0.0, cached=True
            ))
            return self.tracker._cache[ck]

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, image_part],
            config=config,
        )

        text_out = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        cost = _estimate_cost(self.model, in_tok, out_tok)

        self.tracker.calls.append(CallRecord(
            provider=self.provider, model=self.model,
            input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=cost, cached=False
        ))
        self.tracker._cache[ck] = text_out
        return text_out

    def pdf_vision(
        self,
        prompt: str,
        pdf_path: str,
        system_instruction: str = "",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        cache_key: Optional[str] = None,
    ) -> str:
        """Send PDF for multimodal analysis via Gemini."""
        import pathlib
        pdf_bytes = pathlib.Path(pdf_path).read_bytes()
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        ck = cache_key or _hash_key(self.model, system_instruction, prompt, file_hash)
        if ck in self.tracker._cache:
            self.tracker.calls.append(CallRecord(
                provider=self.provider, model=self.model,
                input_tokens=0, output_tokens=0, cost_usd=0.0, cached=True
            ))
            return self.tracker._cache[ck]

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, pdf_part],
            config=config,
        )

        text_out = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        cost = _estimate_cost(self.model, in_tok, out_tok)

        self.tracker.calls.append(CallRecord(
            provider=self.provider, model=self.model,
            input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=cost, cached=False
        ))
        self.tracker._cache[ck] = text_out
        return text_out
