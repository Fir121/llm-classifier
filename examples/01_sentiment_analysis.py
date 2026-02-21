"""
Example 1: Sentiment Analysis with Few-Shot Learning
=====================================================

Dataset: 12 hand-picked Amazon-style product reviews across three sentiment classes.
Source: Adapted from commonly used NLP benchmark examples (no download required).

What this demonstrates:
- Defining a Pydantic output schema with a Literal type
- Providing few-shot examples to steer the model
- Requesting per-prediction reasoning and confidence
- Running batch predictions with parallel execution
- Measuring accuracy against ground-truth labels

Set your API key before running:
    export OPENAI_API_KEY="sk-..."   # Linux / macOS
    $env:OPENAI_API_KEY="sk-..."     # Windows PowerShell
"""

from typing import Literal

from pydantic import BaseModel

from llm_classifier import LLMClassifier

# ---------------------------------------------------------------------------
# 1. Output schema
# ---------------------------------------------------------------------------


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]


# ---------------------------------------------------------------------------
# 2. Inline dataset — reviews with ground-truth labels
# ---------------------------------------------------------------------------

DATASET: list[tuple[str, str]] = [
    # (review text, ground-truth label)
    ("Absolutely love this product! Works exactly as described and arrived on time.", "positive"),
    ("Completely useless. Broke after two days and the seller ignored my refund request.", "negative"),
    ("It's okay. Does what it says but nothing special about it.", "neutral"),
    ("Best purchase I've made this year. The build quality is outstanding!", "positive"),
    ("Very disappointed. The colour looked nothing like the pictures.", "negative"),
    ("Works fine for what I need. Not amazing, not terrible.", "neutral"),
    ("Five stars — exceeded every expectation. Will definitely buy again.", "positive"),
    ("Arrived damaged and customer support was no help whatsoever.", "negative"),
    ("Decent quality for the price. I expected a little more but it does the job.", "neutral"),
    ("Incredible! My whole family loves it. Highly recommend to everyone.", "positive"),
    ("Junk. Stopped working after a week. Total waste of money.", "negative"),
    ("Pretty average. Gets the job done but there are better options out there.", "neutral"),
]

reviews = [text for text, _ in DATASET]
ground_truth = [label for _, label in DATASET]

# ---------------------------------------------------------------------------
# 3. Few-shot examples — shown to the model before each prediction
# ---------------------------------------------------------------------------

FEW_SHOT: list[tuple[str, Sentiment]] = [
    ("This is the best thing I have ever bought, works perfectly!", Sentiment(label="positive")),
    ("Do not buy this. It fell apart immediately and the company does not care.", Sentiment(label="negative")),
    ("It is functional but nothing to write home about.", Sentiment(label="neutral")),
]

# ---------------------------------------------------------------------------
# 4. Run batch prediction
# ---------------------------------------------------------------------------

clf = LLMClassifier(model="openai/gpt-4.1-mini")

print("Running sentiment analysis on 12 reviews...\n")

batch = clf.batch_predict(
    inputs=reviews,
    output_schema=Sentiment,
    examples=FEW_SHOT,
    reasoning=True,
    confidence=True,
    parallel=True,
    max_parallel=6,
)

# ---------------------------------------------------------------------------
# 5. Print results
# ---------------------------------------------------------------------------

correct = 0
print(f"{'#':<3} {'Predicted':<10} {'Actual':<10} {'Conf':>5}  Reasoning snippet")
print("-" * 90)

for i, result in enumerate(batch.results):
    if result is None:
        idx, err = next((e for e in batch.errors if e[0] == i), (i, "unknown"))
        print(f"{i+1:<3} {'ERROR':<10} {ground_truth[i]:<10}  —  {err}")
        continue

    predicted = result.value.label
    actual = ground_truth[i]
    conf = result.confidence or 0.0
    match = "✓" if predicted == actual else "✗"
    snippet = (result.reasoning or "")[:60].replace("\n", " ")

    if predicted == actual:
        correct += 1

    print(f"{i+1:<3} {predicted:<10} {actual:<10} {conf:>4.2f}  {match}  {snippet}…")

print("-" * 90)
accuracy = correct / len(DATASET)
print(f"\nAccuracy: {correct}/{len(DATASET)} = {accuracy:.0%}")
print(f"Successes: {batch.successes}  |  Failures: {batch.failures}")

"""
REAL OUTPUT SNIPPET

Running sentiment analysis on 12 reviews...

Warning: No cache_dir or cache_key provided. Batch caching is disabled, and can cause issue in case execution is interrupted or if you want to reuse results later. To enable caching, set cache_dir and optionally cache_key.
#   Predicted  Actual      Conf  Reasoning snippet
------------------------------------------------------------------------------------------
1   positive   positive   0.95  ✓  The statement expresses strong positive feelings about the p…
2   negative   negative   0.95  ✓  The statement expresses dissatisfaction with the product bei…
3   neutral    neutral    0.90  ✓  The input expresses a neutral sentiment by stating that the …
4   positive   positive   0.95  ✓  The phrase 'Best purchase I've made this year' clearly indic…
5   negative   negative   0.95  ✓  The user expresses clear dissatisfaction with the product, s…
6   neutral    neutral    0.75  ✓  The statement expresses a functional satisfaction with the p…
7   positive   positive   0.95  ✓  The user gave a "Five stars" rating, which is typically the …
8   negative   negative   0.95  ✓  The review states that the item arrived damaged and that cus…
9   neutral    neutral    0.70  ✓  The statement acknowledges that the quality is decent for th…
10  positive   positive   0.95  ✓  The statement uses very positive words such as 'Incredible!'…
11  negative   negative   0.95  ✓  The user review is strongly negative. Words like 'Junk', 'St…
12  neutral    neutral    0.90  ✓  The sentence conveys a neutral sentiment. It indicates that …
------------------------------------------------------------------------------------------

Accuracy: 12/12 = 100%
Successes: 12  |  Failures: 0

REAL OUTPUT SNIPPET RUN 2 (Values have varied in confidence - as expected with non-deterministic models):
Running sentiment analysis on 12 reviews...

Warning: No cache_dir or cache_key provided. Batch caching is disabled, and can cause issue in case execution is interrupted or if you want to reuse results later. To enable caching, set cache_dir and optionally cache_key.
#   Predicted  Actual      Conf  Reasoning snippet
------------------------------------------------------------------------------------------
1   positive   positive   0.95  ✓  The user expresses a strong positive emotion with the phrase…
2   negative   negative   0.95  ✓  The text expresses frustration and dissatisfaction with a pr…
3   neutral    neutral    0.90  ✓  The phrase "It's okay" indicates a neutral sentiment, as it …
4   positive   positive   0.95  ✓  The input expresses a clear positive sentiment by stating it…
5   negative   negative   0.95  ✓  The user expresses disappointment explicitly at the beginnin…
6   neutral    neutral    0.85  ✓  The sentence expresses a moderate sentiment, neither strongl…
7   positive   positive   0.95  ✓  The input message expresses strong positive sentiment. The p…
8   negative   negative   0.95  ✓  The statement indicates a negative experience: the product a…
9   neutral    neutral    0.70  ✓  The user expresses that the quality is decent and acceptable…
10  positive   positive   0.95  ✓  The statement indicates a very positive sentiment as the use…
11  negative   negative   0.95  ✓  The text expresses clear dissatisfaction with the product, s…
12  neutral    neutral    0.75  ✓  The phrase 'Pretty average' indicates a neutral stance. 'Get…
------------------------------------------------------------------------------------------

Accuracy: 12/12 = 100%
Successes: 12  |  Failures: 0
"""
