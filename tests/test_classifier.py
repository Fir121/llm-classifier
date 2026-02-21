"""Tests for llm-classifier package."""

from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_classifier import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    BatchResult,
    LLMClassifier,
    PredictResult,
    create_wrapped_model,
    unwrap_response,
)


# Test schemas
class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]


class Rating(BaseModel):
    score: int
    explanation: str


class MultiLabel(BaseModel):
    tags: list[str]
    primary_tag: str


# ============================================================================
# Wrapper Tests
# ============================================================================

class TestWrappers:
    def test_no_wrapping_when_not_needed(self):
        """When no reasoning/confidence, should return original schema."""
        result = create_wrapped_model(Sentiment, include_reasoning=False, include_confidence=False)
        assert result is Sentiment

    def test_adds_reasoning_field(self):
        """Should add reasoning field when requested."""
        wrapped = create_wrapped_model(Sentiment, include_reasoning=True, include_confidence=False)
        assert "reasoning" in wrapped.model_fields
        assert "label" in wrapped.model_fields  # User schema fields are flattened
        assert wrapped.__name__ == "Sentiment"

    def test_adds_confidence_field(self):
        """Should add confidence field when requested."""
        wrapped = create_wrapped_model(Sentiment, include_reasoning=False, include_confidence=True)
        assert "confidence" in wrapped.model_fields
        assert "label" in wrapped.model_fields  # User schema fields are flattened

    def test_adds_both_fields(self):
        """Should add both fields when both requested."""
        wrapped = create_wrapped_model(Sentiment, include_reasoning=True, include_confidence=True)
        assert "reasoning" in wrapped.model_fields
        assert "confidence" in wrapped.model_fields
        assert "label" in wrapped.model_fields  # User schema fields are flattened

    def test_unwrap_no_wrapping(self):
        """Should return original response when no wrapping."""
        sentiment = Sentiment(label="positive")
        value, reasoning, confidence = unwrap_response(sentiment, Sentiment, False, False)
        assert value is sentiment
        assert reasoning is None
        assert confidence is None

    def test_unwrap_with_reasoning(self):
        """Should extract reasoning from wrapped response."""
        wrapped = create_wrapped_model(Sentiment, include_reasoning=True, include_confidence=False)
        response = wrapped(reasoning="Because...", label="positive")
        value, reasoning, confidence = unwrap_response(response, Sentiment, True, False)
        assert value.label == "positive"
        assert reasoning == "Because..."
        assert confidence is None

    def test_unwrap_with_confidence(self):
        """Should extract confidence from wrapped response."""
        wrapped = create_wrapped_model(Sentiment, include_reasoning=False, include_confidence=True)
        response = wrapped(confidence=0.95, label="negative")
        value, reasoning, confidence = unwrap_response(response, Sentiment, False, True)
        assert value.label == "negative"
        assert reasoning is None
        assert confidence == 0.95


# ============================================================================
# Prompt Building Tests
# ============================================================================

class TestPromptBuilding:
    @pytest.fixture
    def mock_classifier(self):
        """Create classifier with mocked client."""
        with patch("llm_classifier.classifier.instructor.from_provider") as mock:
            mock.return_value = MagicMock()
            return LLMClassifier(model="openai/gpt-4o")

    def test_format_examples_empty(self, mock_classifier):
        """Should return empty string for no examples."""
        result = mock_classifier._format_examples(None)
        assert result == ""

        result = mock_classifier._format_examples([])
        assert result == ""

    def test_format_examples_single(self, mock_classifier):
        """Should format single example correctly."""
        examples = [("I love it", Sentiment(label="positive"))]
        result = mock_classifier._format_examples(examples)
        assert "I love it" in result
        assert "positive" in result
        assert "1." in result

    def test_format_examples_multiple(self, mock_classifier):
        """Should format multiple examples with numbering."""
        examples = [
            ("I love it", Sentiment(label="positive")),
            ("I hate it", Sentiment(label="negative")),
        ]
        result = mock_classifier._format_examples(examples)
        assert "1." in result
        assert "2." in result
        assert "I love it" in result
        assert "I hate it" in result

    def test_build_prompt_includes_input(self, mock_classifier):
        """User prompt should include the input text."""
        system, user = mock_classifier._build_prompt(
            "Test input", Sentiment,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            user_prompt=DEFAULT_USER_PROMPT,
        )
        assert "Test input" in user

    def test_build_prompt_includes_schema(self, mock_classifier):
        """System prompt should include JSON schema."""
        system, user = mock_classifier._build_prompt(
            "Test", Sentiment,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            user_prompt=DEFAULT_USER_PROMPT,
        )
        assert "label" in system
        assert "positive" in system or "negative" in system

    def test_build_prompt_includes_examples(self, mock_classifier):
        """System prompt should include examples when provided."""
        examples = [("Great!", Sentiment(label="positive"))]
        system, user = mock_classifier._build_prompt(
            "Test", Sentiment, examples,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            user_prompt=DEFAULT_USER_PROMPT,
        )
        assert "Great!" in system
        assert "Examples" in system


# ============================================================================
# Result Types Tests
# ============================================================================

class TestResultTypes:
    def test_predict_result_basic(self):
        """PredictResult should hold basic values."""
        result = PredictResult(value=Sentiment(label="positive"))
        assert result.value.label == "positive"
        assert result.reasoning is None
        assert result.confidence is None
        assert result.compliant_variants is None

    def test_predict_result_with_all_fields(self):
        """PredictResult should hold all optional fields."""
        compliant_variants = [
            Sentiment(label="positive"),
            Sentiment(label="positive"),
            Sentiment(label="negative"),
        ]
        result = PredictResult(
            value=Sentiment(label="positive"),
            reasoning="Clear positive language",
            confidence=0.9,
            compliant_variants=compliant_variants,
        )
        assert result.reasoning == "Clear positive language"
        assert result.confidence == 0.9
        assert len(result.compliant_variants) == 3

    def test_batch_result_values_helper(self):
        """BatchResult.values() should extract just the values."""
        results = [
            PredictResult(value=Sentiment(label="positive")),
            None,  # Failed prediction
            PredictResult(value=Sentiment(label="negative")),
        ]
        batch = BatchResult(results=results, successes=2, failures=1)
        values = batch.values()
        assert values[0].label == "positive"
        assert values[1] is None
        assert values[2].label == "negative"


# ============================================================================
# Integration Tests (with mocked LLM)
# ============================================================================

class TestPredict:
    @pytest.fixture
    def mock_classifier(self):
        """Create classifier with mocked Instructor client."""
        with patch("llm_classifier.classifier.instructor.from_provider") as mock_provider:
            mock_client = MagicMock()
            mock_provider.return_value = mock_client
            clf = LLMClassifier(model="openai/gpt-4o")
            return clf, mock_client

    def test_predict_basic(self, mock_classifier):
        """Basic prediction should work."""
        clf, mock_client = mock_classifier
        mock_client.chat.completions.create.return_value = Sentiment(label="positive")

        result = clf.predict("I love this!", output_schema=Sentiment)

        assert result.value.label == "positive"
        mock_client.chat.completions.create.assert_called_once()

    def test_predict_with_reasoning(self, mock_classifier):
        """Prediction with reasoning should include reasoning in result."""
        clf, mock_client = mock_classifier

        # Create wrapped response
        wrapped = create_wrapped_model(Sentiment, include_reasoning=True)
        mock_response = wrapped(
            reasoning="Positive language detected",
            label="positive",
        )
        mock_client.chat.completions.create.return_value = mock_response

        result = clf.predict("I love this!", output_schema=Sentiment, reasoning=True)

        assert result.value.label == "positive"
        assert result.reasoning == "Positive language detected"

    def test_predict_with_confidence(self, mock_classifier):
        """Prediction with confidence should include confidence in result."""
        clf, mock_client = mock_classifier

        wrapped = create_wrapped_model(Sentiment, include_confidence=True)
        mock_response = wrapped(confidence=0.95, label="positive")
        mock_client.chat.completions.create.return_value = mock_response

        result = clf.predict("I love this!", output_schema=Sentiment, confidence=True)

        assert result.value.label == "positive"
        assert result.confidence == 0.95

    def test_predict_consensus_majority(self, mock_classifier):
        """Consensus should return majority vote."""
        clf, mock_client = mock_classifier

        # Mock returns: positive, positive, negative -> majority is positive
        responses = [
            Sentiment(label="positive"),
            Sentiment(label="positive"),
            Sentiment(label="negative"),
        ]
        mock_client.chat.completions.create.side_effect = responses

        result = clf.predict(
            "Somewhat good",
            output_schema=Sentiment,
            consensus=3,
            consensus_parallel=False,  # Sequential for predictable mock behavior
        )

        assert result.value.label == "positive"
        assert len(result.compliant_variants) == 2  # Two positives match majority
        assert len(result.noncompliant_variants) == 1  # One negative is non-compliant
        assert mock_client.chat.completions.create.call_count == 3

    def test_predict_passes_llm_kwargs(self, mock_classifier):
        """LLM kwargs should be passed through."""
        clf, mock_client = mock_classifier
        mock_client.chat.completions.create.return_value = Sentiment(label="neutral")

        clf.predict("Test", output_schema=Sentiment, temperature=0.5, max_tokens=100)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    def test_predict_rejects_invalid_consensus(self, mock_classifier):
        """Predict should fail fast for invalid consensus values."""
        clf, _ = mock_classifier

        with pytest.raises(ValueError, match="consensus must be >= 1"):
            clf.predict("Test", output_schema=Sentiment, consensus=0)

    def test_predict_rejects_invalid_max_parallel(self, mock_classifier):
        """Predict should fail fast for invalid max_parallel values."""
        clf, _ = mock_classifier

        with pytest.raises(ValueError, match="max_parallel must be >= 1"):
            clf.predict("Test", output_schema=Sentiment, max_parallel=0)

    def test_predict_rejects_empty_prompts(self, mock_classifier):
        """Predict should fail when both prompts resolve to empty."""
        clf, _ = mock_classifier

        with pytest.raises(ValueError, match="No prompt messages were generated"):
            clf.predict(
                "Test",
                output_schema=Sentiment,
                system_prompt="",
                user_prompt="",
                validate_placeholders=False,
            )

    def test_predict_consensus_tie_uses_first_seen(self, mock_classifier):
        """On tie, consensus should pick the first seen variant deterministically."""
        clf, mock_client = mock_classifier

        mock_client.chat.completions.create.side_effect = [
            Sentiment(label="positive"),
            Sentiment(label="negative"),
            Sentiment(label="positive"),
            Sentiment(label="negative"),
        ]

        result = clf.predict(
            "Ambiguous sentiment",
            output_schema=Sentiment,
            consensus=4,
            consensus_parallel=False,
        )

        assert result.value.label == "positive"
        assert len(result.compliant_variants) == 2
        assert len(result.noncompliant_variants) == 2


class TestBatchPredict:
    @pytest.fixture
    def mock_classifier(self):
        """Create classifier with mocked Instructor client."""
        with patch("llm_classifier.classifier.instructor.from_provider") as mock_provider:
            mock_client = MagicMock()
            mock_provider.return_value = mock_client
            clf = LLMClassifier(model="openai/gpt-4o")
            return clf, mock_client

    def test_batch_predict_sequential(self, mock_classifier):
        """Batch predict should process all inputs."""
        clf, mock_client = mock_classifier
        mock_client.chat.completions.create.side_effect = [
            Sentiment(label="positive"),
            Sentiment(label="negative"),
            Sentiment(label="neutral"),
        ]

        result = clf.batch_predict(
            ["Good", "Bad", "Okay"],
            output_schema=Sentiment,
            parallel=False,
        )

        assert result.successes == 3
        assert result.failures == 0
        assert len(result.results) == 3
        values = result.values()
        assert values[0].label == "positive"
        assert values[1].label == "negative"
        assert values[2].label == "neutral"

    def test_batch_predict_handles_errors(self, mock_classifier):
        """Batch predict should handle individual failures gracefully."""
        clf, mock_client = mock_classifier
        mock_client.chat.completions.create.side_effect = [
            Sentiment(label="positive"),
            Exception("API Error"),
            Sentiment(label="neutral"),
        ]

        result = clf.batch_predict(
            ["Good", "Bad", "Okay"],
            output_schema=Sentiment,
            parallel=False,
        )

        assert result.successes == 2
        assert result.failures == 1
        assert len(result.errors) == 1
        assert result.errors[0][0] == 1  # Index of failed prediction
        values = result.values()
        assert values[0].label == "positive"
        assert values[1] is None
        assert values[2].label == "neutral"

    def test_batch_predict_rejects_empty_inputs(self, mock_classifier):
        """Batch predict should reject empty input batches."""
        clf, _ = mock_classifier

        with pytest.raises(ValueError, match="inputs must not be empty"):
            clf.batch_predict([], output_schema=Sentiment)

    def test_batch_predict_rejects_invalid_consensus(self, mock_classifier):
        """Batch predict should fail fast for invalid consensus values."""
        clf, _ = mock_classifier

        with pytest.raises(ValueError, match="consensus must be >= 1"):
            clf.batch_predict(["Test"], output_schema=Sentiment, consensus=0)

    def test_batch_predict_rejects_invalid_max_parallel(self, mock_classifier):
        """Batch predict should fail fast for invalid max_parallel values."""
        clf, _ = mock_classifier

        with pytest.raises(ValueError, match="max_parallel must be >= 1"):
            clf.batch_predict(["Test"], output_schema=Sentiment, max_parallel=0)

    def test_batch_predict_resumes_from_cache(self, mock_classifier, tmp_path):
        """Batch predict should skip already-successful cached indices on rerun."""
        clf, mock_client = mock_classifier
        cache_dir = str(tmp_path)

        mock_client.chat.completions.create.side_effect = [
            Sentiment(label="positive"),
            Exception("Transient error"),
            Sentiment(label="neutral"),
        ]

        first = clf.batch_predict(
            ["Good", "Bad", "Okay"],
            output_schema=Sentiment,
            parallel=False,
            cache_dir=cache_dir,
            cache_key="resume_case",
        )

        assert first.successes == 2
        assert first.failures == 1

        mock_client.chat.completions.create.reset_mock()
        mock_client.chat.completions.create.side_effect = [Sentiment(label="negative")]

        second = clf.batch_predict(
            ["Good", "Bad", "Okay"],
            output_schema=Sentiment,
            parallel=False,
            cache_dir=cache_dir,
            cache_key="resume_case",
        )

        assert second.successes == 3
        assert second.failures == 0
        assert mock_client.chat.completions.create.call_count == 1

        values = second.values()
        assert values[0].label == "positive"
        assert values[1].label == "negative"
        assert values[2].label == "neutral"

    def test_batch_predict_cache_per_input_config(self, mock_classifier, tmp_path):
        """Cache should store results per input+config, not per batch."""
        clf, mock_client = mock_classifier
        cache_dir = str(tmp_path)
        mock_client.chat.completions.create.return_value = Sentiment(label="positive")

        # First call with one input
        clf.batch_predict(
            ["Input A"],
            output_schema=Sentiment,
            parallel=False,
            cache_dir=cache_dir,
            cache_key="shared",
        )

        # Second call with different inputs should work fine with same cache_key
        # and should only process "Input B" since "Input A" is cached
        clf.batch_predict(
            ["Input A", "Input B"],
            output_schema=Sentiment,
            parallel=False,
            cache_dir=cache_dir,
            cache_key="shared",
        )

        # Only 2 LLM calls total (Input A once, Input B once)
        assert mock_client.chat.completions.create.call_count == 2

    def test_batch_predict_rejects_cache_key_without_cache_dir(self, mock_classifier):
        """cache_key should require cache_dir."""
        clf, _ = mock_classifier

        with pytest.raises(ValueError, match="cache_key requires cache_dir"):
            clf.batch_predict(
                ["Test"],
                output_schema=Sentiment,
                cache_key="orphan_key",
            )


# ============================================================================
# Custom Schema Tests
# ============================================================================

class TestCustomSchemas:
    @pytest.fixture
    def mock_classifier(self):
        with patch("llm_classifier.classifier.instructor.from_provider") as mock_provider:
            mock_client = MagicMock()
            mock_provider.return_value = mock_client
            clf = LLMClassifier(model="openai/gpt-4o")
            return clf, mock_client

    def test_numeric_schema(self, mock_classifier):
        """Should work with numeric output schemas."""
        clf, mock_client = mock_classifier
        mock_client.chat.completions.create.return_value = Rating(
            score=8, explanation="Good quality"
        )

        result = clf.predict("Rate this product", output_schema=Rating)

        assert result.value.score == 8
        assert result.value.explanation == "Good quality"

    def test_list_schema(self, mock_classifier):
        """Should work with list output schemas."""
        clf, mock_client = mock_classifier
        mock_client.chat.completions.create.return_value = MultiLabel(
            tags=["tech", "news", "ai"],
            primary_tag="tech",
        )

        result = clf.predict("Classify this article", output_schema=MultiLabel)

        assert result.value.tags == ["tech", "news", "ai"]
        assert result.value.primary_tag == "tech"
