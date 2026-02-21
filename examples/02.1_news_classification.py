"""
Example 2: News Headline Classification with Consensus Voting with intermediate cache file for auto resume
=============================================================

Dataset: 16 news headlines inspired by AG News benchmark categories.
Source: Paraphrased/original headlines — no download required.
Categories: world, sports, business, technology

What this demonstrates:
- Same as earlier example but with a cache file

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


class Topic(BaseModel):
    category: Literal["world", "sports", "business", "technology"]


# ---------------------------------------------------------------------------
# 2. Inline dataset — 16 headlines with ground-truth categories
# ---------------------------------------------------------------------------

DATASET: list[tuple[str, str]] = [
    # World
    ("UN Security Council convenes emergency session over escalating border conflict", "world"),
    ("Leaders from 40 nations sign landmark climate agreement at Paris summit", "world"),
    ("Humanitarian aid convoy blocked at border amid ongoing ceasefire talks", "world"),
    ("International observers report irregularities in disputed election results", "world"),
    # Sports
    ("Record-breaking sprinter smashes 100m world record at World Athletics Championships", "sports"),
    ("Underdog squad stuns favourites to claim national football championship", "sports"),
    ("Star midfielder signs €120 million transfer to European club", "sports"),
    ("Host city completes final preparations ahead of Summer Olympics opening ceremony", "sports"),
    # Business
    ("Central bank raises interest rates for the third consecutive quarter", "business"),
    ("Retail giant reports 18% drop in quarterly profits amid slowing consumer demand", "business"),
    ("Merger talks collapse as antitrust regulators raise competition concerns", "business"),
    ("Oil prices surge to two-year high following supply disruption announcement", "business"),
    # Technology
    ("OpenAI announces next-generation model with improved reasoning capabilities", "technology"),
    ("Major data breach exposes personal records of 50 million users", "technology"),
    ("Electric vehicle startup unveils solid-state battery with 800-mile range", "technology"),
    ("Chipmaker unveils new processor architecture designed for AI workloads", "technology"),
]

headlines = [text for text, _ in DATASET]
ground_truth = [label for _, label in DATASET]

# ---------------------------------------------------------------------------
# 3. Run batch prediction with consensus voting
# ---------------------------------------------------------------------------

clf = LLMClassifier(model="openai/gpt-4.1-mini")

print("Classifying 16 news headlines with consensus=5...\n")

batch = clf.batch_predict(
    inputs=headlines,
    output_schema=Topic,
    reasoning=True,
    confidence=True,
    consensus=5,
    consensus_parallel=True,
    parallel=True,
    max_parallel=8,
    cache_dir=".cache"
)

# ---------------------------------------------------------------------------
# 4. Print results
# ---------------------------------------------------------------------------

correct = 0
print(f"{'#':<3} {'Predicted':<12} {'Actual':<12} {'Conf':>5}  {'5-vote split':<20}  Headline")
print("-" * 110)

for i, result in enumerate(batch.results):
    if result is None:
        print(f"{i+1:<3} ERROR        {ground_truth[i]:<12}  —  {headlines[i][:55]}…")
        continue

    predicted = result.value.category
    actual = ground_truth[i]
    conf = result.confidence or 0.0
    match = "✓" if predicted == actual else "✗"

    # Build a readable vote summary from compliant/noncompliant variants
    compliant_count = len(result.compliant_variants or [])
    noncompliant_count = len(result.noncompliant_variants or [])
    vote_str = f"{compliant_count}✓ {noncompliant_count}✗ ({5}/{5})"

    if predicted == actual:
        correct += 1

    snippet = headlines[i][:50]
    print(f"{match} {i+1:<3} {predicted:<12} {actual:<12} {conf:>4.2f}  {vote_str:<20}  {snippet}…")

print("-" * 110)
accuracy = correct / len(DATASET)
print(f"\nAccuracy: {correct}/{len(DATASET)} = {accuracy:.0%}")
print(f"Successes: {batch.successes}  |  Failures: {batch.failures}")

# ---------------------------------------------------------------------------
# Detailed ambiguity report — show headlines where votes were split
# ---------------------------------------------------------------------------

print("\nAmbiguous predictions (noncompliant variants > 0):")
found_any = False
for i, result in enumerate(batch.results):
    if result is None:
        continue
    noncompliant = result.noncompliant_variants or []
    if noncompliant:
        found_any = True
        noncompliant_labels = [v.category for v in noncompliant]
        print(f"  [{i+1}] \"{headlines[i][:65]}…\"")
        print(f"       Chosen: {result.value.category!r} | Dissenting: {noncompliant_labels}")

if not found_any:
    print("  (all predictions were unanimous)")

"""
REAL OUTPUT SNIPPET

Classifying 16 news headlines with consensus=5...

#   Predicted    Actual        Conf  5-vote split          Headline
--------------------------------------------------------------------------------------------------------------
✓ 1   world        world        0.90  5✓ 0✗ (5/5)           UN Security Council convenes emergency session ove…
✓ 2   world        world        0.95  5✓ 0✗ (5/5)           Leaders from 40 nations sign landmark climate agre…
✓ 3   world        world        0.85  5✓ 0✗ (5/5)           Humanitarian aid convoy blocked at border amid ong…
✓ 4   world        world        0.85  5✓ 0✗ (5/5)           International observers report irregularities in d…
✓ 5   sports       sports       0.95  5✓ 0✗ (5/5)           Record-breaking sprinter smashes 100m world record…
✓ 6   sports       sports       0.95  5✓ 0✗ (5/5)           Underdog squad stuns favourites to claim national …
✓ 7   sports       sports       0.95  5✓ 0✗ (5/5)           Star midfielder signs €120 million transfer to Eur…
✓ 8   sports       sports       0.95  5✓ 0✗ (5/5)           Host city completes final preparations ahead of Su…
✓ 9   business     business     0.95  5✓ 0✗ (5/5)           Central bank raises interest rates for the third c…
✓ 10  business     business     0.95  5✓ 0✗ (5/5)           Retail giant reports 18% drop in quarterly profits…
✓ 11  business     business     0.95  5✓ 0✗ (5/5)           Merger talks collapse as antitrust regulators rais…
✓ 12  business     business     0.95  5✓ 0✗ (5/5)           Oil prices surge to two-year high following supply…
✓ 13  technology   technology   0.95  5✓ 0✗ (5/5)           OpenAI announces next-generation model with improv…
✓ 14  technology   technology   0.90  5✓ 0✗ (5/5)           Major data breach exposes personal records of 50 m…
✓ 15  technology   technology   0.95  5✓ 0✗ (5/5)           Electric vehicle startup unveils solid-state batte…
✓ 16  technology   technology   0.90  5✓ 0✗ (5/5)           Chipmaker unveils new processor architecture desig…
--------------------------------------------------------------------------------------------------------------

Accuracy: 16/16 = 100%
Successes: 16  |  Failures: 0

Ambiguous predictions (noncompliant variants > 0):
  (all predictions were unanimous)

** NOTE: Cache file created and results saved to .cache/input_cache.jsonl for potential reuse or inspection. **
"""
