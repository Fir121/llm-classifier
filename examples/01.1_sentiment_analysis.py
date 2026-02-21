"""
Example 1.1: Sentiment Analysis with Zero Shot but customizing prompts for fine tuning to the task
=====================================================

Dataset: 12 hand-picked Amazon-style product reviews across three sentiment classes.
Source: Adapted from commonly used NLP benchmark examples (no download required).

What this demonstrates:
- Same as earlier example but without few-shot examples, instead customizing the system and user prompts to steer the model

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
# 3. Custom prompts for tasks with no few-shot examples (zero-shot)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful and precise assistant for classifying the sentiment of product reviews. "
    "Given a review, you will determine if the sentiment is positive, negative, or neutral. "
    "A positive review expresses strong satisfaction or enthusiasm. "
    "A negative review expresses dissatisfaction or frustration. "
    "A neutral review expresses a balanced or indifferent sentiment. "
    "Your output schema:\n{format}"
)

USER_PROMPT = "Classify the sentiment of the following review:\n\n\"{input}\""

# ---------------------------------------------------------------------------
# 4. Run batch prediction
# ---------------------------------------------------------------------------

clf = LLMClassifier(model="openai/gpt-4.1-mini")

print("Running sentiment analysis on 12 reviews...\n")

batch = clf.batch_predict(
    inputs=reviews,
    output_schema=Sentiment,
    system_prompt=SYSTEM_PROMPT,
    user_prompt=USER_PROMPT,
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
1   positive   positive   0.95  ✓  The review expresses strong positive language such as 'Absol…
2   negative   negative   0.95  ✓  The review expresses strong dissatisfaction with the product…
3   neutral    neutral    0.90  ✓  The review states "It's okay," which shows a neutral stance,…
4   positive   positive   0.95  ✓  The review expresses strong positive sentiment by stating "B…
5   negative   negative   0.95  ✓  The review expresses dissatisfaction explicitly by saying "V…
6   neutral    neutral    0.95  ✓  The review expresses a neutral sentiment. The phrases 'Works…
7   positive   positive   0.95  ✓  The review expresses strong satisfaction and enthusiasm by s…
8   negative   negative   0.95  ✓  The review expresses dissatisfaction with two aspects: the p…
9   neutral    neutral    0.85  ✓  The review expresses a balanced sentiment. The reviewer ackn…
10  positive   positive   0.95  ✓  The review uses strong positive words and phrases such as 'I…
11  negative   negative   0.95  ✓  The review clearly expresses dissatisfaction by calling the …
12  neutral    neutral    0.90  ✓  The review uses the phrase "Pretty average," indicating a ne…
------------------------------------------------------------------------------------------

Accuracy: 12/12 = 100%
Successes: 12  |  Failures: 0
"""
