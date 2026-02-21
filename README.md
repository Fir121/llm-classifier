# llm-classifier

Structured classification and extraction on top of [Instructor](https://github.com/jxnl/instructor), with typed outputs via Pydantic.

## Why use it

- Return validated Pydantic models instead of free-form text
- Add few-shot examples directly in each call
- Optionally collect `reasoning` and `confidence`
- Reduce variance with consensus voting
- Run batched predictions with per-item error capture

## Installation

```bash
pip install llm-classifier
```

## Quickstart

```python
from typing import Literal
from pydantic import BaseModel
from llm_classifier import LLMClassifier


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]


clf = LLMClassifier(model="openai/gpt-4o")

result = clf.predict(
    input="This movie was amazing!",
    output_schema=Sentiment,
    examples=[
        ("I hated it", Sentiment(label="negative")),
        ("It was okay", Sentiment(label="neutral")),
    ],
    reasoning=True,
    confidence=True,
)

print(result.value.label)   # "positive"
print(result.reasoning)     # Optional[str]
print(result.confidence)    # Optional[float]
```

## How it works

```mermaid
flowchart LR
    A[Input text] --> B[Build prompts\n+schema + examples]
    B --> C[LLM call via Instructor]
    C --> D[Structured response\n+Pydantic-validated]
    D --> E[Unwrap into PredictResult]
    E --> F[value]
    E --> G[reasoning optional]
    E --> H[confidence optional]
```

## Core API

### Single prediction

```python
result = clf.predict(
    input="This is somewhat good",
    output_schema=Sentiment,
    consensus=5,
    consensus_parallel=True,
    max_parallel=3,
)

print(result.value)
print(result.compliant_variants)     # Variants matching selected output
print(result.noncompliant_variants)  # Variants not matching selected output
```

### Batch prediction

```python
batch = clf.batch_predict(
    inputs=["Great", "Bad", "Okay"],
    output_schema=Sentiment,
    parallel=True,
    max_parallel=5,
    cache_dir="./.llm_cache",
    cache_key="sentiment_run_2026_02_21",
)

print(batch.successes, batch.failures)
print(batch.values())   # [Sentiment | None, ...]
print(batch.errors)     # [(index, Exception), ...]
```

### Resumable batch cache

When `cache_dir` is set, each processed index is appended to a cache log so reruns skip already successful items.

- Cache file: `<cache_dir>/<cache_key>.jsonl` (defaults to `input_cache.jsonl` when `cache_key` is omitted)
- `cache_key` requires `cache_dir` to also be set, otherwise a `ValueError` is raised

Each `.jsonl` line is one step record keyed by a SHA-256 hash of the full input configuration (model, text, schema, examples, prompts, settings). On rerun with the same `cache_dir` + `cache_key`, already-cached inputs are skipped.

## Clustering with LLMCluster

For bulk clustering of many items in a **single LLM call**, use `LLMCluster`. This is ideal when you have many rows (e.g., 100 survey responses) and want to group them into high-level clusters without making N separate calls.

### Basic usage

```python
from pydantic import BaseModel
from llm_classifier import LLMCluster


class ClusterSchema(BaseModel):
    name: str
    summary: str


clusterer = LLMCluster(model="openai/gpt-4o")

surveys = [
    "The product quality is excellent!",
    "Shipping was too slow",
    "Great customer service",
    "Product broke after one week",
    "Fast delivery, very happy",
    "Support team was unhelpful",
]

# cluster() requires (index, text) tuples — use enumerate to build them
indexed_surveys = list(enumerate(surveys, 1))

result = clusterer.cluster(
    inputs=indexed_surveys,
    cluster_schema=ClusterSchema,
)

for cluster in result.clusters:
    print(f"\n{cluster.cluster.name}: {cluster.cluster.summary}")
    for idx, text in cluster.references:
        print(f"  [{idx}] {text}")
```

### How it works

```mermaid
flowchart LR
    A[N input items] --> B[Auto-assign IDs\n1...N]
    B --> C[Single LLM call]
    C --> D[Structured clusters\nwith item_ids]
    D --> E[Post-hoc validation]
    E -->|Pass| F[ClusterResult]
    E -->|Fail| G[Retry with feedback]
    G --> C
```

Items are assigned numeric IDs `[1]` through `[N]`, and the LLM returns clusters referencing these IDs. Validation ensures referential integrity before returning results.

### Cluster schema

Define a Pydantic model for per-cluster fields. A `reference_ids: list[int]` field is automatically injected at runtime to track which items belong to each cluster — you do not need to add it yourself:

```python
class TopicCluster(BaseModel):
    name: str
    description: str
    sentiment: Literal["positive", "negative", "mixed"]
```

### Validation and retries

The clusterer validates LLM responses and retries on failures:

| Check | Behavior |
|-------|----------|
| Invalid ID (outside 1..N) | Always fails |
| Duplicate ID across clusters | Fails when `allow_overlap=False` (default) |
| Missing ID (item not in any cluster) | Fails when `require_all=True` (default) |
| Empty cluster | Always fails |

```python
result = clusterer.cluster(
    inputs=list(enumerate(surveys, 1)),
    cluster_schema=ClusterSchema,
    allow_overlap=False,     # Each item in exactly one cluster
    require_all=True,        # Every item must be assigned
    max_retries=3,           # Instructor retries for malformed JSON
    validation_retries=2,    # Our retries for referential integrity errors
)

print(f"Validation retries used: {result.retries_used}")
```

### Cluster count hint

Let the LLM decide the number of clusters, or provide a hint:

```python
# LLM decides
result = clusterer.cluster(inputs=list(enumerate(surveys, 1)), cluster_schema=ClusterSchema)

# Suggest 3 clusters
result = clusterer.cluster(inputs=list(enumerate(surveys, 1)), cluster_schema=ClusterSchema, n_clusters=3)
```

### Error handling

```python
from llm_classifier import ClusterValidationError, ContextLengthError

try:
    result = clusterer.cluster(inputs=list(enumerate(huge_list, 1)), cluster_schema=ClusterSchema)
except ContextLengthError as e:
    print(f"Too many items for model context: {e}")
except ClusterValidationError as e:
    print(f"Validation failed after retries: {e.errors}")
```

## Real-world examples

Runnable scripts are in the [`examples/`](./examples) folder. Each includes inline data so you only need an API key to run them.

### Example 1 — Sentiment analysis with few-shot learning ([`examples/01_sentiment_analysis.py`](./examples/01_sentiment_analysis.py))

12 Amazon-style product reviews → `positive / negative / neutral` using `batch_predict` with 3 few-shot examples, `reasoning=True`, and `confidence=True`.

| # | Review snippet | Predicted | Actual | Conf |
|---|---------------|-----------|--------|------|
| 1 | "Absolutely love this product! Works exactly…" | positive | positive | 0.99 |
| 2 | "Completely useless. Broke after two days…" | negative | negative | 0.99 |
| 3 | "It's okay. Does what it says but nothing special…" | neutral | neutral | 0.90 |
| 4 | "Best purchase I've made this year…" | positive | positive | 0.99 |
| 5 | "Very disappointed. The colour looked nothing…" | negative | negative | 0.97 |
| 6 | "Works fine for what I need. Not amazing, not terrible." | neutral | neutral | 0.88 |
| … | … | … | … | … |

**Result: 12/12 correct (100% accuracy)**

---

### Example 2 — News topic classification with consensus voting ([`examples/02_news_classification.py`](./examples/02_news_classification.py))

16 news headlines → `world / sports / business / technology` using zero-shot `batch_predict` with `consensus=5` (parallel). The 5-vote split shows the model's internal certainty.

| # | Headline snippet | Predicted | Actual | Conf | Vote split |
|---|-----------------|-----------|--------|------|------------|
| 1 | "UN Security Council convenes emergency session…" | world | world | 0.97 | 5✓ 0✗ |
| 5 | "Record-breaking sprinter smashes 100m world record…" | sports | sports | 0.99 | 5✓ 0✗ |
| 9 | "Central bank raises interest rates for the third…" | business | business | 0.97 | 5✓ 0✗ |
| 13 | "OpenAI announces next-generation model…" | technology | technology | 0.99 | 5✓ 0✗ |
| … | … | … | … | … | … |

**Result: 16/16 correct (100% accuracy), all votes unanimous**

---

### Example 3 — Survey response clustering ([`examples/03_survey_clustering.py`](./examples/03_survey_clustering.py))

20 open-ended NPS feedback responses for a SaaS product → 5 named clusters in a **single LLM call** using `LLMCluster`. Each cluster includes a generated name, summary, and sentiment.

| Cluster | Sentiment | # Responses | Theme |
|---------|-----------|-------------|-------|
| Onboarding and Setup | positive | 2 | Easy setup wizard, fast first run |
| Collaboration and Integrations | positive | 4 | Real-time teamwork, Slack/GitHub/Jira |
| Performance and Reliability Issues | negative | 5 | Dashboard slowness, freezes, mobile crashes |
| Pricing Concerns | negative | 2 | Free tier limits, unexpected price hike |
| Feature Requests | mixed | 7 | Reporting, Gantt charts, search, notifications |

**Result: all 20 responses assigned, 0 validation retries needed**

---

## Behavior notes

- `consensus` must be `>= 1`, else `ValueError`
- `max_parallel` must be `>= 1`, else `ValueError`
- `batch_predict(inputs=[])` raises `ValueError`
- If both prompts resolve to empty, prediction raises `ValueError`
- Consensus tie-break is deterministic: first-seen variant wins
- `cache_key` requires `cache_dir`, otherwise a `ValueError` is raised

## Model support

Use any provider/model string supported by `instructor.from_provider(...)`, for example:

- `openai/gpt-4.1`
- `anthropic/claude-3-5-sonnet-20241022`
- `google/gemini-1.5-pro`

## License

Apache-2.0 License. See [LICENSE](./LICENSE) for details.
