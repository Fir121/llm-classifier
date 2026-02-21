"""Tests for LLMCluster clustering functionality."""

from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from llm_classifier import (
    ClusterItem,
    ClusterResult,
    ClusterValidationError,
    ContextLengthError,
    LLMCluster,
)
from llm_classifier.cluster import (
    DEFAULT_CLUSTER_SYSTEM_PROMPT,
    DEFAULT_CLUSTER_USER_PROMPT,
)


# Test schemas
class SimpleCluster(BaseModel):
    name: str


class DetailedCluster(BaseModel):
    name: str
    summary: str
    sentiment: Literal["positive", "negative", "mixed"]


# ============================================================================
# Result Types Tests
# ============================================================================


class TestClusterResultTypes:
    def test_cluster_item_basic(self):
        """ClusterItem should hold cluster data and items."""
        cluster = SimpleCluster(name="Positive Reviews")
        item = ClusterItem(
            cluster=cluster,
            references=[(1, "Great!"), (3, "Love it"), (5, "Amazing")],
        )
        assert item.cluster.name == "Positive Reviews"
        assert item.references == [(1, "Great!"), (3, "Love it"), (5, "Amazing")]
        assert len(item.references) == 3

    def test_cluster_result_basic(self):
        """ClusterResult should hold clusters."""
        cluster1 = ClusterItem(
            cluster=SimpleCluster(name="Group A"),
            references=[(1, "Item 1"), (2, "Item 2")],
        )
        cluster2 = ClusterItem(
            cluster=SimpleCluster(name="Group B"),
            references=[(3, "Item 3")],
        )
        result = ClusterResult(clusters=[cluster1, cluster2])
        assert len(result.clusters) == 2
        assert result.retries_used == 0

    def test_cluster_result_with_all_fields(self):
        """ClusterResult should hold all optional fields."""
        cluster = ClusterItem(
            cluster=SimpleCluster(name="Test"),
            references=[(1, "Item")],
        )
        result = ClusterResult(
            clusters=[cluster],
            retries_used=1,
        )
        assert result.retries_used == 1


# ============================================================================
# Item Formatting Tests
# ============================================================================


class TestItemFormatting:
    @pytest.fixture
    def mock_clusterer(self):
        """Create clusterer with mocked client."""
        with patch("llm_classifier.cluster.instructor.from_provider") as mock:
            mock.return_value = MagicMock()
            return LLMCluster(model="openai/gpt-4o")

    def test_format_items_basic(self, mock_clusterer):
        """Should format items with numeric IDs."""
        items = [(1, "First item"), (2, "Second item"), (3, "Third item")]
        result = mock_clusterer._format_items(items)
        assert "[REFERENCE ID 1] First item" in result
        assert "[REFERENCE ID 2] Second item" in result
        assert "[REFERENCE ID 3] Third item" in result

    def test_format_items_escapes_brackets(self, mock_clusterer):
        """Should escape brackets in item text."""
        items = [(1, "Item with [brackets]")]
        result = mock_clusterer._format_items(items)
        assert "[REFERENCE ID 1] Item with \\[brackets\\]" in result


# ============================================================================
# Schema Building Tests
# ============================================================================


class TestSchemaBuiding:
    @pytest.fixture
    def mock_clusterer(self):
        """Create clusterer with mocked client."""
        with patch("llm_classifier.cluster.instructor.from_provider") as mock:
            mock.return_value = MagicMock()
            return LLMCluster(model="openai/gpt-4o")

    def test_build_schema_adds_reference_ids(self, mock_clusterer):
        """Should add reference_ids field to cluster schema."""
        schema = mock_clusterer._build_cluster_schema(SimpleCluster, n_items=10)
        # The schema should have a "clusters" field containing wrapped clusters
        assert "clusters" in schema.model_fields

    def test_build_schema_rejects_existing_reference_ids(self, mock_clusterer):
        """Should raise error if schema already has reference_ids field."""

        class BadSchema(BaseModel):
            name: str
            reference_ids: list[int]

        with pytest.raises(ValueError, match="reference_ids"):
            mock_clusterer._build_cluster_schema(BadSchema, n_items=10)


# ============================================================================
# Prompt Building Tests
# ============================================================================


class TestPromptBuilding:
    @pytest.fixture
    def mock_clusterer(self):
        """Create clusterer with mocked client."""
        with patch("llm_classifier.cluster.instructor.from_provider") as mock:
            mock.return_value = MagicMock()
            return LLMCluster(model="openai/gpt-4o")

    def test_build_prompt_includes_items(self, mock_clusterer):
        """User prompt should include formatted items."""
        inputs = [(1, "Survey 1"), (2, "Survey 2")]
        response_schema = mock_clusterer._build_cluster_schema(SimpleCluster, n_items=2)
        system, user = mock_clusterer._build_prompt(
            inputs, response_schema, n_clusters=None, allow_overlap=False, require_all=True,
            system_prompt=DEFAULT_CLUSTER_SYSTEM_PROMPT, user_prompt=DEFAULT_CLUSTER_USER_PROMPT
        )
        assert "[REFERENCE ID 1] Survey 1" in user
        assert "[REFERENCE ID 2] Survey 2" in user

    def test_build_prompt_includes_schema(self, mock_clusterer):
        """System prompt should include JSON schema."""
        inputs = [(1, "Test")]
        response_schema = mock_clusterer._build_cluster_schema(SimpleCluster, n_items=1)
        system, user = mock_clusterer._build_prompt(
            inputs, response_schema, n_clusters=None, allow_overlap=False, require_all=True,
            system_prompt=DEFAULT_CLUSTER_SYSTEM_PROMPT, user_prompt=DEFAULT_CLUSTER_USER_PROMPT
        )
        assert "name" in system
        assert "reference_ids" in system

    def test_build_prompt_with_n_clusters(self, mock_clusterer):
        """System prompt should include cluster count hint when provided."""
        inputs = [(1, "Test")]
        response_schema = mock_clusterer._build_cluster_schema(SimpleCluster, n_items=1)
        system, user = mock_clusterer._build_prompt(
            inputs, response_schema, n_clusters=5, allow_overlap=False, require_all=True,
            system_prompt=DEFAULT_CLUSTER_SYSTEM_PROMPT, user_prompt=DEFAULT_CLUSTER_USER_PROMPT
        )
        assert "exactly 5 clusters" in system

    def test_build_prompt_without_n_clusters(self, mock_clusterer):
        """System prompt should mention LLM decides when no hint."""
        inputs = [(1, "Test")]
        response_schema = mock_clusterer._build_cluster_schema(SimpleCluster, n_items=1)
        system, user = mock_clusterer._build_prompt(
            inputs, response_schema, n_clusters=None, allow_overlap=False, require_all=True,
            system_prompt=DEFAULT_CLUSTER_SYSTEM_PROMPT, user_prompt=DEFAULT_CLUSTER_USER_PROMPT
        )
        assert "as many clusters as you see fit" in system

    def test_build_prompt_validation_rules_no_overlap(self, mock_clusterer):
        """System prompt should include no-overlap rule."""
        inputs = [(1, "Test")]
        response_schema = mock_clusterer._build_cluster_schema(SimpleCluster, n_items=1)
        system, user = mock_clusterer._build_prompt(
            inputs, response_schema, n_clusters=None, allow_overlap=False, require_all=True,
            system_prompt=DEFAULT_CLUSTER_SYSTEM_PROMPT, user_prompt=DEFAULT_CLUSTER_USER_PROMPT
        )
        assert "exactly ONE cluster" in system

    def test_build_prompt_validation_rules_require_all(self, mock_clusterer):
        """System prompt should include require-all rule."""
        inputs = [(1, "Test")]
        response_schema = mock_clusterer._build_cluster_schema(SimpleCluster, n_items=1)
        system, user = mock_clusterer._build_prompt(
            inputs, response_schema, n_clusters=None, allow_overlap=False, require_all=True,
            system_prompt=DEFAULT_CLUSTER_SYSTEM_PROMPT, user_prompt=DEFAULT_CLUSTER_USER_PROMPT
        )
        assert "Every reference ID must be assigned" in system


# ============================================================================
# Validation Tests
# ============================================================================


class TestValidation:
    @pytest.fixture
    def mock_clusterer(self):
        """Create clusterer with mocked client."""
        with patch("llm_classifier.cluster.instructor.from_provider") as mock:
            mock.return_value = MagicMock()
            return LLMCluster(model="openai/gpt-4o")

    def test_validate_valid_response(self, mock_clusterer):
        """Valid response should pass validation."""

        class MockCluster:
            reference_ids = [1, 2]

        class MockResponse:
            clusters = [MockCluster()]

        errors = mock_clusterer._validate_clusters(
            MockResponse(), n_items=2, allow_overlap=False, require_all=True
        )
        assert errors == []

    def test_validate_invalid_id(self, mock_clusterer):
        """Invalid ID should fail validation."""

        class MockCluster:
            reference_ids = [1, 99]  # 99 is invalid for n_items=3

        class MockResponse:
            clusters = [MockCluster()]

        errors = mock_clusterer._validate_clusters(
            MockResponse(), n_items=3, allow_overlap=False, require_all=False
        )
        assert any("Invalid reference ID 99" in e for e in errors)

    def test_validate_duplicate_id_no_overlap(self, mock_clusterer):
        """Duplicate ID should fail when overlap not allowed."""

        class MockCluster1:
            reference_ids = [1, 2]

        class MockCluster2:
            reference_ids = [2, 3]  # 2 is duplicate

        class MockResponse:
            clusters = [MockCluster1(), MockCluster2()]

        errors = mock_clusterer._validate_clusters(
            MockResponse(), n_items=3, allow_overlap=False, require_all=True
        )
        assert any("appears in multiple clusters" in e for e in errors)

    def test_validate_duplicate_id_with_overlap(self, mock_clusterer):
        """Duplicate ID should pass when overlap allowed."""

        class MockCluster1:
            reference_ids = [1, 2]

        class MockCluster2:
            reference_ids = [2, 3]  # 2 appears in both

        class MockResponse:
            clusters = [MockCluster1(), MockCluster2()]

        errors = mock_clusterer._validate_clusters(
            MockResponse(), n_items=3, allow_overlap=True, require_all=True
        )
        assert errors == []

    def test_validate_missing_id_require_all(self, mock_clusterer):
        """Missing ID should fail when require_all=True."""

        class MockCluster:
            reference_ids = [1, 3]  # Missing 2

        class MockResponse:
            clusters = [MockCluster()]

        errors = mock_clusterer._validate_clusters(
            MockResponse(), n_items=3, allow_overlap=False, require_all=True
        )
        assert any("not assigned to any cluster" in e for e in errors)

    def test_validate_missing_id_not_required(self, mock_clusterer):
        """Missing ID should pass when require_all=False."""

        class MockCluster:
            reference_ids = [1, 3]  # Missing 2

        class MockResponse:
            clusters = [MockCluster()]

        errors = mock_clusterer._validate_clusters(
            MockResponse(), n_items=3, allow_overlap=False, require_all=False
        )
        assert errors == []

    def test_validate_empty_cluster(self, mock_clusterer):
        """Empty cluster should fail validation."""

        class MockCluster:
            reference_ids = []

        class MockResponse:
            clusters = [MockCluster()]

        errors = mock_clusterer._validate_clusters(
            MockResponse(), n_items=3, allow_overlap=False, require_all=False
        )
        assert any("has no references assigned" in e for e in errors)


# ============================================================================
# Integration Tests (with mocked LLM)
# ============================================================================


class TestCluster:
    @pytest.fixture
    def mock_clusterer(self):
        """Create clusterer with mocked Instructor client."""
        with patch("llm_classifier.cluster.instructor.from_provider") as mock_provider:
            mock_client = MagicMock()
            mock_provider.return_value = mock_client
            clusterer = LLMCluster(model="openai/gpt-4o")
            return clusterer, mock_client

    def _create_mock_response(self, clusters_data):
        """Helper to create mock LLM responses."""

        class MockCluster:
            def __init__(self, name, reference_ids):
                self.name = name
                self.reference_ids = reference_ids

        class MockResponse:
            def __init__(self):
                self.clusters = [
                    MockCluster(c["name"], c["reference_ids"]) for c in clusters_data
                ]

        return MockResponse()

    def test_cluster_basic(self, mock_clusterer):
        """Basic clustering should work."""
        clusterer, mock_client = mock_clusterer

        mock_client.chat.completions.create.return_value = self._create_mock_response(
            [
                {"name": "Positive", "reference_ids": [1, 3]},
                {"name": "Negative", "reference_ids": [2]},
            ]
        )

        result = clusterer.cluster(
            inputs=[(1, "Great!"), (2, "Terrible"), (3, "Amazing")],
            cluster_schema=SimpleCluster,
        )

        assert len(result.clusters) == 2
        assert result.clusters[0].cluster.name == "Positive"
        assert result.clusters[0].references == [(1, "Great!"), (3, "Amazing")]
        assert result.clusters[1].cluster.name == "Negative"
        assert result.clusters[1].references == [(2, "Terrible")]
        mock_client.chat.completions.create.assert_called_once()

    def test_cluster_validates_empty_inputs(self, mock_clusterer):
        """Should reject empty inputs."""
        clusterer, _ = mock_clusterer

        with pytest.raises(ValueError, match="inputs cannot be empty"):
            clusterer.cluster(inputs=[], cluster_schema=SimpleCluster)

    def test_cluster_validates_n_clusters(self, mock_clusterer):
        """Should reject invalid n_clusters."""
        clusterer, _ = mock_clusterer

        with pytest.raises(ValueError, match="n_clusters must be at least 1"):
            clusterer.cluster(
                inputs=[(1, "Test")], cluster_schema=SimpleCluster, n_clusters=0
            )

    def test_cluster_rejects_empty_prompts(self, mock_clusterer):
        """Should fail when both prompts resolve to empty."""
        clusterer, _ = mock_clusterer

        with pytest.raises(ValueError, match="No prompt messages were generated"):
            clusterer.cluster(
                inputs=[(1, "Test")],
                cluster_schema=SimpleCluster,
                system_prompt="",
                user_prompt="",
                validate_placeholders=False,
            )

    def test_cluster_passes_llm_kwargs(self, mock_clusterer):
        """LLM kwargs should be passed through."""
        clusterer, mock_client = mock_clusterer

        mock_client.chat.completions.create.return_value = self._create_mock_response(
            [{"name": "All", "reference_ids": [1]}]
        )

        clusterer.cluster(
            inputs=[(1, "Test")],
            cluster_schema=SimpleCluster,
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100


class TestClusterValidationRetries:
    @pytest.fixture
    def mock_clusterer(self):
        """Create clusterer with mocked Instructor client."""
        with patch("llm_classifier.cluster.instructor.from_provider") as mock_provider:
            mock_client = MagicMock()
            mock_provider.return_value = mock_client
            clusterer = LLMCluster(model="openai/gpt-4o")
            return clusterer, mock_client

    def _create_mock_response(self, clusters_data):
        """Helper to create mock LLM responses."""

        class MockCluster:
            def __init__(self, name, reference_ids):
                self.name = name
                self.reference_ids = reference_ids

        class MockResponse:
            def __init__(self):
                self.clusters = [
                    MockCluster(c["name"], c["reference_ids"]) for c in clusters_data
                ]

        return MockResponse()

    def test_validation_retry_on_invalid_id(self, mock_clusterer):
        """Should retry when LLM returns invalid ID."""
        clusterer, mock_client = mock_clusterer

        # First response has invalid ID, second is valid
        bad_response = self._create_mock_response(
            [{"name": "All", "reference_ids": [1, 99]}]  # 99 is invalid
        )
        good_response = self._create_mock_response([{"name": "All", "reference_ids": [1, 2]}])

        mock_client.chat.completions.create.side_effect = [bad_response, good_response]

        result = clusterer.cluster(
            inputs=[(1, "Item 1"), (2, "Item 2")],
            cluster_schema=SimpleCluster,
            validation_retries=2,
        )

        assert result.retries_used == 1
        assert mock_client.chat.completions.create.call_count == 2

    def test_validation_retry_on_duplicate_id(self, mock_clusterer):
        """Should retry when LLM returns duplicate IDs (overlap not allowed)."""
        clusterer, mock_client = mock_clusterer

        # First response has duplicate, second is valid
        bad_response = self._create_mock_response(
            [
                {"name": "A", "reference_ids": [1, 2]},
                {"name": "B", "reference_ids": [2, 3]},  # 2 is duplicate
            ]
        )
        good_response = self._create_mock_response(
            [
                {"name": "A", "reference_ids": [1, 2]},
                {"name": "B", "reference_ids": [3]},
            ]
        )

        mock_client.chat.completions.create.side_effect = [bad_response, good_response]

        result = clusterer.cluster(
            inputs=[(1, "Item 1"), (2, "Item 2"), (3, "Item 3")],
            cluster_schema=SimpleCluster,
            allow_overlap=False,
            validation_retries=2,
        )

        assert result.retries_used == 1

    def test_validation_retry_on_missing_id(self, mock_clusterer):
        """Should retry when LLM misses an ID (require_all=True)."""
        clusterer, mock_client = mock_clusterer

        # First response misses ID 2, second is valid
        bad_response = self._create_mock_response(
            [{"name": "Partial", "reference_ids": [1, 3]}]  # Missing 2
        )
        good_response = self._create_mock_response(
            [{"name": "All", "reference_ids": [1, 2, 3]}]
        )

        mock_client.chat.completions.create.side_effect = [bad_response, good_response]

        result = clusterer.cluster(
            inputs=[(1, "Item 1"), (2, "Item 2"), (3, "Item 3")],
            cluster_schema=SimpleCluster,
            require_all=True,
            validation_retries=2,
        )

        assert result.retries_used == 1

    def test_validation_error_after_all_retries(self, mock_clusterer):
        """Should raise ClusterValidationError after all retries exhausted."""
        clusterer, mock_client = mock_clusterer

        # All responses are invalid
        bad_response = self._create_mock_response(
            [{"name": "Bad", "reference_ids": [1, 99]}]  # 99 is invalid
        )

        mock_client.chat.completions.create.return_value = bad_response

        with pytest.raises(ClusterValidationError) as exc_info:
            clusterer.cluster(
                inputs=[(1, "Item 1"), (2, "Item 2")],
                cluster_schema=SimpleCluster,
                validation_retries=2,
            )

        assert "Invalid reference ID 99" in str(exc_info.value)
        assert len(exc_info.value.errors) > 0
        # Should have called: initial + 2 retries = 3 times
        assert mock_client.chat.completions.create.call_count == 3

    def test_duplicate_allowed_with_overlap(self, mock_clusterer):
        """Should not retry when duplicates allowed (allow_overlap=True)."""
        clusterer, mock_client = mock_clusterer

        # Response has item in multiple clusters
        response = self._create_mock_response(
            [
                {"name": "A", "reference_ids": [1, 2]},
                {"name": "B", "reference_ids": [2, 3]},  # 2 is in both
            ]
        )

        mock_client.chat.completions.create.return_value = response

        result = clusterer.cluster(
            inputs=[(1, "Item 1"), (2, "Item 2"), (3, "Item 3")],
            cluster_schema=SimpleCluster,
            allow_overlap=True,  # Overlap is allowed
            require_all=True,
        )

        assert result.retries_used == 0
        mock_client.chat.completions.create.assert_called_once()

    def test_missing_allowed_when_not_required(self, mock_clusterer):
        """Should not retry when missing IDs allowed (require_all=False)."""
        clusterer, mock_client = mock_clusterer

        # Response misses item 2
        response = self._create_mock_response(
            [{"name": "Partial", "reference_ids": [1, 3]}]  # Missing 2
        )

        mock_client.chat.completions.create.return_value = response

        result = clusterer.cluster(
            inputs=[(1, "Item 1"), (2, "Item 2"), (3, "Item 3")],
            cluster_schema=SimpleCluster,
            require_all=False,  # Missing items allowed
        )

        assert result.retries_used == 0
        mock_client.chat.completions.create.assert_called_once()


class TestContextLengthError:
    @pytest.fixture
    def mock_clusterer(self):
        """Create clusterer with mocked Instructor client."""
        with patch("llm_classifier.cluster.instructor.from_provider") as mock_provider:
            mock_client = MagicMock()
            mock_provider.return_value = mock_client
            clusterer = LLMCluster(model="openai/gpt-4o")
            return clusterer, mock_client

    def test_context_length_error_detection(self, mock_clusterer):
        """Should raise ContextLengthError on context length API errors."""
        clusterer, mock_client = mock_clusterer

        mock_client.chat.completions.create.side_effect = Exception(
            "This model's maximum context length is 128000 tokens"
        )

        with pytest.raises(ContextLengthError) as exc_info:
            clusterer.cluster(
                inputs=[(i, "Item " * 1000) for i in range(1, 101)],  # Lots of data
                cluster_schema=SimpleCluster,
            )

        assert "context window" in str(exc_info.value).lower()
        assert exc_info.value.original_exception is not None

    def test_non_context_error_propagates(self, mock_clusterer):
        """Non-context errors should propagate normally."""
        clusterer, mock_client = mock_clusterer

        mock_client.chat.completions.create.side_effect = ValueError("Some other error")

        with pytest.raises(ValueError, match="Some other error"):
            clusterer.cluster(inputs=[(1, "Test")], cluster_schema=SimpleCluster)


class TestCustomPrompts:
    @pytest.fixture
    def mock_clusterer(self):
        """Create clusterer with mocked Instructor client."""
        with patch("llm_classifier.cluster.instructor.from_provider") as mock_provider:
            mock_client = MagicMock()
            mock_provider.return_value = mock_client
            clusterer = LLMCluster(model="openai/gpt-4o")
            return clusterer, mock_client

    def _create_mock_response(self, clusters_data):
        """Helper to create mock LLM responses."""

        class MockCluster:
            def __init__(self, name, reference_ids):
                self.name = name
                self.reference_ids = reference_ids

        class MockResponse:
            def __init__(self):
                self.clusters = [
                    MockCluster(c["name"], c["reference_ids"]) for c in clusters_data
                ]

        return MockResponse()

    def test_custom_system_prompt_at_init(self, mock_clusterer):
        """Custom system prompt passed to cluster() should be used."""
        with patch("llm_classifier.cluster.instructor.from_provider") as mock_provider:
            mock_client = MagicMock()
            mock_provider.return_value = mock_client

            custom_prompt = "Custom system: {format} {n_clusters_instruction} {validation_rules}"
            clusterer = LLMCluster(model="openai/gpt-4o")

            mock_client.chat.completions.create.return_value = self._create_mock_response(
                [{"name": "All", "reference_ids": [1]}]
            )

            clusterer.cluster(
                inputs=[(1, "Test")],
                cluster_schema=SimpleCluster,
                system_prompt=custom_prompt,
            )

            # Check that custom prompt was used
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            system_msg = next(m for m in messages if m["role"] == "system")
            assert "Custom system:" in system_msg["content"]

    def test_custom_user_prompt_at_call(self, mock_clusterer):
        """Custom user prompt at call should override default."""
        clusterer, mock_client = mock_clusterer

        mock_client.chat.completions.create.return_value = self._create_mock_response(
            [{"name": "All", "reference_ids": [1]}]
        )

        clusterer.cluster(
            inputs=[(1, "Test")],
            cluster_schema=SimpleCluster,
            user_prompt="Custom user: {items}",
        )

        # Check that custom prompt was used
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Custom user:" in user_msg["content"]

    def test_only_system_prompt_mode(self):
        """only_system_prompt should combine prompts."""
        with patch("llm_classifier.cluster.instructor.from_provider") as mock_provider:
            mock_client = MagicMock()
            mock_provider.return_value = mock_client

            clusterer = LLMCluster(model="openai/gpt-4o")

            class MockCluster:
                name = "All"
                reference_ids = [1]

            class MockResponse:
                clusters = [MockCluster()]

            mock_client.chat.completions.create.return_value = MockResponse()

            clusterer.cluster(
                inputs=[(1, "Test")],
                cluster_schema=SimpleCluster,
                only_system_prompt=True,
            )

            # Should have only system message
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "system"
            # Should contain both system and user content
            assert "[REFERENCE ID 1] Test" in messages[0]["content"]


class TestExceptions:
    def test_cluster_validation_error_attributes(self):
        """ClusterValidationError should have errors attribute."""
        errors = ["Error 1", "Error 2"]
        exc = ClusterValidationError(errors)
        assert exc.errors == errors
        assert "Error 1" in str(exc)
        assert "Error 2" in str(exc)

    def test_cluster_validation_error_custom_message(self):
        """ClusterValidationError should accept custom message."""
        exc = ClusterValidationError(["Error"], message="Custom message")
        assert str(exc) == "Custom message"
        assert exc.errors == ["Error"]

    def test_context_length_error_attributes(self):
        """ContextLengthError should have original_exception attribute."""
        original = ValueError("Original error")
        exc = ContextLengthError(original)
        assert exc.original_exception is original
        assert "Original error" in str(exc)
        assert "context window" in str(exc).lower()
