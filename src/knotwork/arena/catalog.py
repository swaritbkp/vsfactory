"""Vision model catalog: OpenRouter's live list of image-input chat models, with a curated offline
fallback and per-token USD pricing.

The live list is fetched only when a person asks for it (`fetch(live=True)`); any failure --
offline, network policy, timeout, malformed reply -- silently returns the curated catalog, so the
page never breaks without a network. Curated prices are planning estimates, labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from knotwork import net

MODELS_URL = (
    "https://openrouter.ai/api/v1/models"
    "?input_modalities=image&endpoints=/api/v1/chat/completions&sort=most-popular"
)
TIMEOUT_S = 6.0
ROUTING_SUFFIXES = (":nitro", ":floor", ":exacto", ":online", ":thinking", ":beta")
#: the committed live benchmark's three models first: they are the default comparison
DEFAULT_MODELS = (
    "openai/gpt-5.4",
    "google/gemini-2.5-flash",
    "qwen/qwen3-vl-235b-a22b-instruct",
)


@dataclass(frozen=True)
class CatalogModel:
    """A model and its price in USD per million tokens (prompt, completion)."""

    id: str
    name: str
    prompt_per_m: float
    completion_per_m: float
    context: int = 0

    @property
    def free(self) -> bool:
        return self.id.endswith(":free") or (self.prompt_per_m == 0 and self.completion_per_m == 0)

    @property
    def label(self) -> str:
        if self.free:
            return f"{self.id} · free"
        return f"{self.id} · ${self.prompt_per_m:g} / ${self.completion_per_m:g} per M"


CURATED: tuple[CatalogModel, ...] = (
    CatalogModel("openai/gpt-5.4", "OpenAI GPT-5.4", 2.50, 15.00, 400_000),
    CatalogModel("google/gemini-2.5-flash", "Google Gemini 2.5 Flash", 0.30, 2.50, 1_048_576),
    CatalogModel("qwen/qwen3-vl-235b-a22b-instruct", "Qwen3-VL 235B", 0.30, 1.20, 262_144),
    CatalogModel("google/gemini-2.5-pro", "Google Gemini 2.5 Pro", 1.25, 10.00, 1_048_576),
    CatalogModel(
        "anthropic/claude-sonnet-4.5", "Anthropic Claude Sonnet 4.5", 3.00, 15.00, 200_000
    ),
    CatalogModel("openai/gpt-5-mini", "OpenAI GPT-5 mini", 0.25, 2.00, 400_000),
    CatalogModel("openai/gpt-4.1-mini", "OpenAI GPT-4.1 mini", 0.40, 1.60, 1_047_576),
    CatalogModel("meta-llama/llama-4-maverick", "Meta Llama 4 Maverick", 0.15, 0.60, 1_048_576),
    CatalogModel("mistralai/mistral-medium-3.1", "Mistral Medium 3.1", 0.40, 2.00, 131_072),
    CatalogModel("qwen/qwen2.5-vl-72b-instruct", "Qwen2.5-VL 72B", 0.25, 0.75, 32_768),
    CatalogModel("google/gemma-3-27b-it:free", "Google Gemma 3 27B (free)", 0.0, 0.0, 96_000),
    CatalogModel(
        "mistralai/mistral-small-3.2-24b-instruct:free",
        "Mistral Small 3.2 (free)",
        0.0,
        0.0,
        96_000,
    ),
)


@dataclass(frozen=True)
class Catalog:
    models: tuple[CatalogModel, ...]
    source: str  # "live" | "curated"
    note: str = ""

    def ids(self, free_only: bool = False) -> list[str]:
        return [m.id for m in self.models if m.free or not free_only]

    def get(self, model_id: str) -> CatalogModel | None:
        base = base_id(model_id)
        return next((m for m in self.models if m.id in (model_id, base)), None)


def base_id(model_id: str) -> str:
    """The catalog id behind a routing suffix (`:nitro`, `:floor`, ...); `:free` is a distinct id."""
    for suffix in ROUTING_SUFFIXES:
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def curated() -> Catalog:
    return Catalog(CURATED, "curated", "Curated offline catalog (planning prices).")


def parse_models(data: Any) -> tuple[CatalogModel, ...]:
    """OpenRouter `/models` JSON -> models with image input; prices there are USD per token."""
    rows = data.get("data", []) if isinstance(data, dict) else []
    out = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        modalities = (row.get("architecture") or {}).get("input_modalities") or ["image"]
        if "image" not in modalities:
            continue
        price = row.get("pricing") or {}
        try:
            prompt = float(price.get("prompt") or 0.0) * 1e6
            completion = float(price.get("completion") or 0.0) * 1e6
        except (TypeError, ValueError):
            continue
        if prompt < 0 or completion < 0:  # "-1" marks variable-priced routers
            continue
        out.append(
            CatalogModel(
                str(row["id"]),
                str(row.get("name") or row["id"]),
                round(prompt, 6),
                round(completion, 6),
                int(row.get("context_length") or 0),
            )
        )
    return tuple(out)


def fetch(live: bool = False, transport: Any = None) -> Catalog:
    """The live catalog when ``live`` and reachable, else the curated one (never raises)."""
    if not live:
        return curated()
    try:
        data = net.get_json(MODELS_URL, TIMEOUT_S, transport, policy_override="openrouter")
        models = parse_models(data)
    except Exception as exc:  # offline, policy, timeout, bad JSON: fall back quietly
        return Catalog(CURATED, "curated", f"Live catalog unavailable ({type(exc).__name__}).")
    if not models:
        return Catalog(CURATED, "curated", "Live catalog returned no vision models.")
    return Catalog(models, "live", f"{len(models)} vision models from OpenRouter.")


def prices(model_id: str, catalog: Catalog | None = None) -> tuple[float, float]:
    """(USD per prompt token, USD per completion token); unknown models use a cautious default."""
    if model_id.endswith(":free"):
        return 0.0, 0.0
    found = (catalog or curated()).get(model_id) or curated().get(model_id)
    if found is None:
        return 3.0e-6, 15.0e-6  # unknown model: priced like a frontier model, never under-budgeted
    return found.prompt_per_m / 1e6, found.completion_per_m / 1e6


def call_cost(
    model_id: str, prompt_tokens: int, completion_tokens: int, catalog: Catalog | None = None
) -> float:
    p, c = prices(model_id, catalog)
    return prompt_tokens * p + completion_tokens * c
