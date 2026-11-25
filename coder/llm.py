"""LLM providers, cost accounting and the budget guard.

Budget rule from the brief: one US$1 budget for the whole run, across both
providers; a call that would pass it is refused upstream. We never want to get
there, so every call *reserves* its worst-case cost (estimated input tokens +
``max_tokens`` of output) before it is sent, and settles to the real usage
reported by the API afterwards. If a reservation would cross
``CODER_BUDGET_USD`` (default 0.90, leaving headroom for estimation error), the
call is not made and the caller falls back to the offline answer.

Networking: the OpenAI and Anthropic Python SDKs read ``HTTPS_PROXY`` from the
environment by default. We create clients with default HTTP settings so that
behaviour is kept. Nothing else talks to the network at run time.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time

# US$ per 1M tokens (input, output). Unknown models get a pessimistic price so
# the guard errs on the side of stopping early.
PRICES = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-haiku-5-5": (1.00, 5.00),
}
UNKNOWN_PRICE = (3.00, 15.00)

OPENAI_MODELS = [m.strip() for m in os.environ.get(
    "CODER_OPENAI_MODELS", "gpt-4.1-mini,gpt-4o-mini,gpt-5-mini").split(",") if m.strip()]
ANTHROPIC_MODELS = [m.strip() for m in os.environ.get(
    "CODER_ANTHROPIC_MODELS", "claude-haiku-4-5,claude-haiku-5-5,claude-3-5-haiku-latest").split(",") if m.strip()]


def log(msg: str) -> None:
    print(f"[coder] {msg}", file=sys.stderr, flush=True)


def price(model: str) -> tuple[float, float]:
    for k, v in PRICES.items():
        if model == k or model.startswith(k + "-20"):  # dated snapshots
            return v
    return UNKNOWN_PRICE


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (non-Latin scripts cost more per char)."""
    return int(max(len(text) / 3.0, len(text.encode("utf-8")) / 3.2)) + 16


class BudgetExceeded(Exception):
    pass


class ModelUnavailable(Exception):
    pass


class Budget:
    def __init__(self, limit_usd: float):
        self.limit = limit_usd
        self.spent = 0.0
        self.reserved = 0.0
        self.calls = 0
        self.refused = 0
        self._lock = threading.Lock()

    def reserve(self, usd: float) -> None:
        with self._lock:
            if self.spent + self.reserved + usd > self.limit:
                self.refused += 1
                raise BudgetExceeded(f"spent {self.spent:.4f} + reserved {self.reserved:.4f} + {usd:.4f} > {self.limit}")
            self.reserved += usd

    def settle(self, reserved: float, actual: float) -> None:
        with self._lock:
            self.reserved -= reserved
            self.spent += actual
            self.calls += 1

    def remaining(self) -> float:
        with self._lock:
            return self.limit - self.spent - self.reserved


class Provider:
    """One vendor, with an ordered list of models to fall back through."""

    name = "base"

    def __init__(self, models: list[str], budget: Budget, timeout: float = 40.0):
        self.models = list(models)
        self.model = self.models[0]
        self.budget = budget
        self.timeout = timeout
        self._lock = threading.Lock()
        self.dead = False
        self.consecutive_failures = 0
        self.max_consecutive_failures = 12

    # subclasses implement _call(model, system, user, max_tokens) -> (text, in_tok, out_tok)
    def _call(self, model, system, user, max_tokens):  # pragma: no cover
        raise NotImplementedError

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        if self.dead:
            raise ModelUnavailable("provider disabled")
        while True:
            model = self.model
            pin, pout = price(model)
            est = (estimate_tokens(system + user) * pin + max_tokens * pout) / 1e6
            self.budget.reserve(est)
            try:
                text, tin, tout = self._call(model, system, user, max_tokens)
            except ModelUnavailable as e:
                self.budget.settle(est, 0.0)
                self._next_model(model, e)
                continue
            except Exception as e:
                # A request that never left (connection/proxy error) cost nothing;
                # anything else (timeout, 5xx) may have been billed, so count it.
                no_cost = ("connection" in type(e).__name__.lower()
                           or isinstance(e, (TypeError, ValueError, AttributeError)))  # failed locally
                self.budget.settle(est, 0.0 if no_cost else est)
                self._failure(e)
                raise
            actual = (tin * pin + tout * pout) / 1e6 if (tin or tout) else est
            self.budget.settle(est, actual)
            with self._lock:
                self.consecutive_failures = 0
            return text

    def _failure(self, e: Exception) -> None:
        """Circuit breaker: if the API keeps failing (bad key, no proxy), stop
        trying so the remaining notes finish quickly on the offline path."""
        with self._lock:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures and not self.dead:
                self.dead = True
                log(f"{self.name}: {self.consecutive_failures} failures in a row, disabling LLM "
                    f"(last: {type(e).__name__}: {str(e)[:160]})")

    def _next_model(self, failed: str, err: Exception) -> None:
        with self._lock:
            if self.model != failed:
                return  # another thread already moved on
            i = self.models.index(failed)
            if i + 1 >= len(self.models):
                self.dead = True
                log(f"{self.name}: no usable model ({err})")
                raise ModelUnavailable(str(err))
            self.model = self.models[i + 1]
            log(f"{self.name}: model {failed} unavailable ({str(err)[:120]}); trying {self.model}")


def _looks_like_bad_model(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}".lower()
    return ("notfound" in s or "404" in s or "model_not_found" in s or "does not exist" in s
            or ("model" in s and ("not supported" in s or "invalid" in s or "unknown" in s)))


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from openai import OpenAI
        self.client = OpenAI(timeout=self.timeout, max_retries=2)

    def _call(self, model, system, user, max_tokens):
        reasoning = model.startswith(("gpt-5", "o1", "o3", "o4"))
        kw = dict(model=model,
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                  response_format={"type": "json_object"})
        if reasoning:
            kw["max_completion_tokens"] = max_tokens + 400  # room for minimal reasoning
            kw["reasoning_effort"] = "minimal"
        else:
            kw["max_tokens"] = max_tokens
            kw["temperature"] = 0
        try:
            r = self.client.chat.completions.create(**kw)
        except Exception as e:
            if _looks_like_bad_model(e):
                raise ModelUnavailable(str(e)) from e
            raise
        u = getattr(r, "usage", None)
        return (r.choices[0].message.content or "",
                getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        import anthropic
        self.client = anthropic.Anthropic(timeout=self.timeout, max_retries=2)

    def _call(self, model, system, user, max_tokens):
        try:
            # Note: the pinned SDK (anthropic 2.x) no longer takes ``temperature``.
            r = self.client.messages.create(model=model, max_tokens=max_tokens,
                                            system=system, messages=[{"role": "user", "content": user}])
        except Exception as e:
            if _looks_like_bad_model(e):
                raise ModelUnavailable(str(e)) from e
            raise
        text = "".join(getattr(b, "text", "") for b in r.content)
        u = getattr(r, "usage", None)
        return text, getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0


def _sanitize_tls_env() -> None:
    """Drop CA-bundle variables that point at files that do not exist.

    A host path leaking into the container (e.g. SSL_CERT_FILE=/home/me/ca.crt)
    makes the SDKs fail at client creation, which would silently disable the
    LLM for the whole run.
    """
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        v = os.environ.get(var)
        if v and not os.path.exists(v):
            log(f"ignoring {var}={v} (file not found)")
            os.environ.pop(var, None)


def make_provider(budget: Budget) -> Provider | None:
    """OpenAI if its key is set (cheaper per note), else Anthropic, else None.

    ``CODER_PROVIDER=openai|anthropic|none`` overrides the choice.
    """
    want = os.environ.get("CODER_PROVIDER", "auto").lower()
    has_oa = bool(os.environ.get("OPENAI_API_KEY"))
    has_an = bool(os.environ.get("ANTHROPIC_API_KEY"))
    _sanitize_tls_env()
    order = {"openai": ["openai"], "anthropic": ["anthropic"], "none": []}.get(want, ["openai", "anthropic"])
    for name in order:
        try:
            if name == "openai" and has_oa:
                return OpenAIProvider(OPENAI_MODELS, budget)
            if name == "anthropic" and has_an:
                return AnthropicProvider(ANTHROPIC_MODELS, budget)
        except Exception as e:  # SDK missing or misconfigured
            log(f"could not start {name} client: {e}")
    return None


_JSON = re.compile(r"\{.*\}", re.S)


def parse_json(text: str) -> dict:
    """Parse a JSON object out of a model reply (tolerates code fences / prose)."""
    try:
        return json.loads(text)
    except Exception:
        m = _JSON.search(text or "")
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def retry_call(fn, attempts: int = 2, base_sleep: float = 1.5):
    """Retry transient failures (rate limits, timeouts); never retry budget/model errors."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as e:
            last = e
            time.sleep(base_sleep * (i + 1))
    raise last
