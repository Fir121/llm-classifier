"""
Example 3: Clustering Open-Ended Survey Responses
==================================================

Dataset: 20 fictional NPS (Net Promoter Score) open-ended survey responses
for a SaaS project management tool. No download required.

What this demonstrates:
- Using LLMCluster to group many short texts in a single LLM call
- Defining a rich per-cluster schema with sentiment
- Requesting a specific number of clusters via n_clusters
- Inspecting ClusterResult.retries_used to see if validation retries were needed
- Correct inputs format: list[tuple[int, str]] using enumerate()

Set your API key before running:
    export OPENAI_API_KEY="sk-..."   # Linux / macOS
    $env:OPENAI_API_KEY="sk-..."     # Windows PowerShell
"""

from typing import Literal

from pydantic import BaseModel

from llm_classifier import ClusterValidationError, ContextLengthError, LLMCluster

# ---------------------------------------------------------------------------
# 1. Per-cluster schema — fields the LLM fills in for each discovered group
# ---------------------------------------------------------------------------


class FeedbackCluster(BaseModel):
    name: str
    summary: str
    sentiment: Literal["positive", "negative", "mixed"]


# ---------------------------------------------------------------------------
# 2. Inline dataset — 20 open-ended SaaS product survey responses
# ---------------------------------------------------------------------------

RESPONSES: list[str] = [
    # Positive — onboarding
    "Getting started was surprisingly smooth. The setup wizard is excellent.",
    "I was up and running in under 10 minutes. Great first experience.",
    # Positive — collaboration
    "The real-time collaboration features are top-notch. My team loves it.",
    "Assigning tasks and commenting on them in one place saves us hours every week.",
    # Positive — integrations
    "Slack and GitHub integrations work flawlessly. Saves so much context switching.",
    "The Jira import was painless. All our existing issues transferred perfectly.",
    # Negative — performance
    "The dashboard takes forever to load when we have more than 20 projects.",
    "It freezes on me at least twice a day. Really slows down my workflow.",
    "Why does switching between projects take 5 seconds? Fix the performance please.",
    # Negative — mobile app
    "The mobile app is basically unusable. Crashes constantly on iOS.",
    "Cannot edit tasks from my phone at all. Please invest in the mobile experience.",
    # Negative — pricing
    "The free tier is too limited. I hit the project cap after one week.",
    "Pricing jumped by 40% at renewal without much notice. Considering alternatives.",
    # Neutral / feature request — reporting
    "Reporting could be much better. I need burndown charts and velocity tracking.",
    "Would love Gantt chart support. Right now I have to export everything to Excel.",
    # Neutral / feature request — notifications
    "Too many email notifications by default. Need better granular controls.",
    "There's no way to mute a project temporarily. That would be really useful.",
    # Neutral / feature request — search
    "Search is hard to use. I can never find old tasks buried in closed projects.",
    "Full-text search across comments would make this tool perfect for us.",
    # General positive
    "Overall really solid product. A few rough edges but we renew every year.",
]

# ---------------------------------------------------------------------------
# 3. Build LLMCluster and cluster the responses
#    NOTE: inputs must be list[tuple[int, str]] — use enumerate() to build them
# ---------------------------------------------------------------------------

clusterer = LLMCluster(model="openai/gpt-4.1")

indexed_responses = list(enumerate(RESPONSES, 1))

print(f"Clustering {len(RESPONSES)} survey responses into ~5 groups...\n")

try:
    result = clusterer.cluster(
        inputs=indexed_responses,
        cluster_schema=FeedbackCluster,
        n_clusters=5,          # hint: we want roughly 5 high-level themes
        allow_overlap=False,   # each response goes in exactly one cluster
        require_all=True,      # every response must be assigned
        validation_retries=2,  # retry up to 2 times if referential integrity fails
    )
except ContextLengthError as e:
    print(f"Input too large for model context: {e}")
    raise SystemExit(1)
except ClusterValidationError as e:
    print(f"Clustering failed after retries: {e.errors}")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# 4. Print results
# ---------------------------------------------------------------------------

print(f"Validation retries used: {result.retries_used}\n")
print("=" * 70)

for cluster_item in result.clusters:
    c = cluster_item.cluster
    emoji = {"positive": "✅", "negative": "❌", "mixed": "🔀"}.get(c.sentiment, "")
    print(f"\n{emoji}  {c.name.upper()}  [{c.sentiment}]")
    print(f"   {c.summary}")
    print(f"   Responses ({len(cluster_item.references)}):")
    for idx, text in cluster_item.references:
        print(f"     [{idx:>2}] {text}")

print("\n" + "=" * 70)
total_assigned = sum(len(ci.references) for ci in result.clusters)
print(f"\nTotal clusters: {len(result.clusters)}")
print(f"Total responses assigned: {total_assigned}/{len(RESPONSES)}")

"""
REAL OUTPUT SNIPPET

Clustering 20 survey responses into ~5 groups...

Validation retries used: 0

======================================================================

✅  SMOOTH ONBOARDING AND SETUP EXPERIENCE  [positive]
   Comments praising the onboarding process, setup wizard, and ease of starting use.
   Responses (3):
     [ 1] Getting started was surprisingly smooth. The setup wizard is excellent.
     [ 2] I was up and running in under 10 minutes. Great first experience.
     [ 6] The Jira import was painless. All our existing issues transferred perfectly.

✅  COLLABORATION AND INTEGRATION FEATURES  [positive]
   Positive feedback about collaboration tools and integrations with other products.
   Responses (3):
     [ 3] The real-time collaboration features are top-notch. My team loves it.
     [ 4] Assigning tasks and commenting on them in one place saves us hours every week.
     [ 5] Slack and GitHub integrations work flawlessly. Saves so much context switching.

❌  PERFORMANCE AND USABILITY ISSUES  [negative]
   Complaints regarding performance, speed, freezing, and issues with the mobile app.
   Responses (5):
     [ 7] The dashboard takes forever to load when we have more than 20 projects.
     [ 8] It freezes on me at least twice a day. Really slows down my workflow.
     [ 9] Why does switching between projects take 5 seconds? Fix the performance please.
     [10] The mobile app is basically unusable. Crashes constantly on iOS.
     [11] Cannot edit tasks from my phone at all. Please invest in the mobile experience.

❌  PRICING AND LIMITATIONS  [negative]
   Critiques about the cost, changes in pricing, and product limitations.
   Responses (2):
     [12] The free tier is too limited. I hit the project cap after one week.
     [13] Pricing jumped by 40% at renewal without much notice. Considering alternatives.

🔀  FEATURE REQUESTS AND PRODUCT SUGGESTIONS  [mixed]
   Requests for additional features, better reporting, search improvements, and notification controls.
   Responses (7):
     [14] Reporting could be much better. I need burndown charts and velocity tracking.
     [15] Would love Gantt chart support. Right now I have to export everything to Excel.
     [16] Too many email notifications by default. Need better granular controls.
     [17] There's no way to mute a project temporarily. That would be really useful.
     [18] Search is hard to use. I can never find old tasks buried in closed projects.
     [19] Full-text search across comments would make this tool perfect for us.
     [20] Overall really solid product. A few rough edges but we renew every year.

======================================================================

Total clusters: 5
Total responses assigned: 20/20
"""
