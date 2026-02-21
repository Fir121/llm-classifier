"""Core LLMCluster implementation."""

import json
from dataclasses import dataclass
from typing import Any, Generic, Type, TypeVar

import instructor
from pydantic import BaseModel, Field, create_model

T = TypeVar("T", bound=BaseModel)


# ============================================================================
# Result Types
# ============================================================================


@dataclass
class ClusterItem(Generic[T]):
    """A single cluster with its assigned items.

    Attributes:
        cluster: The cluster data matching the user's schema (e.g., name, summary)
        references: List of (index, text) tuples for items assigned to this cluster
    """

    cluster: T
    references: list[tuple[int, str]]


@dataclass
class ClusterResult(Generic[T]):
    """Result of a clustering operation.

    Attributes:
        clusters: List of ClusterItem containing grouped items
        raw_response: The raw response from the LLM
        retries_used: Number of validation retries that were needed
    """

    clusters: list[ClusterItem[T]]
    raw_response: BaseModel | None = None
    retries_used: int = 0


# ============================================================================
# Prompts and Exceptions
# ============================================================================

DEFAULT_CLUSTER_SYSTEM_PROMPT = """You are a clustering assistant. Group items into clusters.

Each item has a numeric ID like [1], [2]. Reference items ONLY by their IDs.

{n_clusters_instruction}

Output schema:
{format}

CRITICAL RULES:
- Reference items ONLY by their numeric IDs (integers)
- Each cluster must have at least one item
{validation_rules}"""

DEFAULT_CLUSTER_USER_PROMPT = """Cluster the following items:

{items}"""


class ClusterValidationError(ValueError):
    """Raised when cluster validation fails after all retries are exhausted.

    Attributes:
        errors: List of specific validation error messages
    """

    def __init__(self, errors: list[str], message: str | None = None):
        self.errors = errors
        msg = message or f"Cluster validation failed: {'; '.join(errors)}"
        super().__init__(msg)


class ContextLengthError(ValueError):
    """Raised when the input exceeds the model's context window.

    Attributes:
        original_exception: The underlying API exception
    """

    def __init__(self, original_exception: Exception):
        self.original_exception = original_exception
        super().__init__(
            f"Input too large for model context window. "
            f"Try reducing the number of items or using a model with a larger context. "
            f"Original error: {original_exception}"
        )


class LLMCluster:
    """LLM-based clustering with structured outputs.

    Groups multiple items into clusters using a single LLM call. Each item is
    assigned a numeric ID (1 to N), and the LLM returns clusters referencing
    these IDs. Includes validation and retry logic for referential integrity.

    Args:
        model: Model identifier in format "provider/model-name"
            Examples: "openai/gpt-4.1", "anthropic/claude-4-6-sonnet"
        api_key: Optional API key (defaults to environment variables)
        **client_kwargs: Additional kwargs passed to the Instructor client

    Example:
        >>> from pydantic import BaseModel
        >>>
        >>> class ClusterSchema(BaseModel):
        ...     name: str
        ...     summary: str
        >>>
        >>> clusterer = LLMCluster(model="openai/gpt-4o")
        >>> surveys = ["Great product!", "Terrible service", "Love it", "Needs work"]
        >>> result = clusterer.cluster(surveys, cluster_schema=ClusterSchema)
        >>> for cluster in result.clusters:
        ...     print(f"{cluster.cluster.name}: {cluster.items}")
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        **client_kwargs: Any,
    ):
        self.model = model
        self.api_key = api_key
        self.client_kwargs = client_kwargs

        # Initialize instructor client
        if api_key:
            self._client = instructor.from_provider(model, api_key=api_key, **client_kwargs)
        else:
            self._client = instructor.from_provider(model, **client_kwargs)

    def _format_items(self, inputs: list[tuple[int, str]]) -> str:
        """Format input items with numeric IDs for the prompt.

        Args:
            inputs: List of text items to cluster

        Returns:
            Formatted string with items numbered [1] through [N]
        """
        formatted = []
        for i, item in enumerate(inputs, 1):
            # Escape any brackets in the item text to avoid confusion
            escaped = item[1].replace("[", "\\[").replace("]", "\\]")
            formatted.append(f"[REFERENCE ID {i}] {escaped}")
        return "\n".join(formatted)

    def _build_cluster_schema(
        self,
        cluster_schema: Type[T],
        n_items: int
    ) -> Type[BaseModel]:
        """Build the response schema with reference_ids field injected.

        Args:
            cluster_schema: User's per-cluster Pydantic model
            n_items: Number of items (for field description)

        Returns:
            A new Pydantic model for the full clustering response

        Raises:
            ValueError: If cluster_schema already has 'reference_ids' field
            as this is reserved for tracking item references
        """
        # Check for conflicting field names
        if "reference_ids" in cluster_schema.model_fields:
            raise ValueError("Cluster schema already has a 'reference_ids' field")

        # Build the per-cluster model with reference_ids injected
        cluster_fields: dict = {}

        # Add all user-defined fields first
        for field_name, field_info in cluster_schema.model_fields.items():
            cluster_fields[field_name] = (field_info.annotation, field_info)

        # Add reference_ids field
        cluster_fields["reference_ids"] = (
            list[int],
            Field(
                description=(
                    f"List of Reference IDs (integers 1 to {n_items}) "
                    "belonging to this cluster"
                )
            ),
        )

        # Create the wrapped cluster model
        WrappedCluster = create_model(
            f"{cluster_schema.__name__}WithIds",
            **cluster_fields,
        )

        # Build the response model fields
        response_fields: dict = {}


        response_fields["clusters"] = (
            list[WrappedCluster],
            Field(description="List of clusters with their assigned reference IDs"),
        )

        # Create the full response model
        ClusterResponse = create_model("ClusterResponse", **response_fields)

        return ClusterResponse

    def _build_validation_rules(
        self,
        allow_overlap: bool,
        require_all: bool,
    ) -> str:
        """Build validation rules text for the prompt."""
        rules = []

        if not allow_overlap:
            rules.append(
                "- Each reference ID must appear in exactly ONE cluster (no duplicates)"
            )

        if require_all:
            rules.append(
                "- Every reference ID must be assigned to a cluster (none left out)"
            )

        if rules:
            return "\n".join(rules)
        return "- References may appear in multiple clusters or be left unclustered"

    def _build_n_clusters_instruction(self, n_clusters: int | str | None) -> str:
        """Build the cluster count instruction for the prompt."""
        if isinstance(n_clusters, int):
            return f"Create exactly {n_clusters} clusters."
        elif isinstance(n_clusters, str):
            return f"Create {n_clusters} clusters."
        return "Create as many clusters as you see fit based on natural groupings in the data."

    def _build_prompt(
        self,
        inputs: list[tuple[int, str]],
        response_schema: Type[BaseModel],
        n_clusters: int | str | None,
        allow_overlap: bool,
        require_all: bool,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        validate_placeholders: bool = True,
    ) -> tuple[str | None, str | None]:
        """Build the system and user prompts from templates and components."""
        items_str = self._format_items(inputs)

        format_schema = json.dumps(response_schema.model_json_schema(), indent=2)

        n_clusters_instruction = self._build_n_clusters_instruction(n_clusters)
        validation_rules = self._build_validation_rules(allow_overlap, require_all)

        # Raise an error if any placeholder has a value but appears in neither prompt.
        if validate_placeholders:
            params = [
                ("{format}", format_schema),
                ("{n_clusters_instruction}", n_clusters_instruction),
                ("{validation_rules}", validation_rules),
                ("{items}", items_str),
            ]
            for placeholder, value in params:
                if (
                    value
                    and (system_prompt is None or placeholder not in system_prompt)
                    and (user_prompt is None or placeholder not in user_prompt)
                ):
                    raise ValueError(
                        f"Placeholder {placeholder} has a value but is missing from both prompts. "
                        "To suppress the error, set validate_placeholders=False."
                    )

        # Build system prompt (or None if falsy)
        if not system_prompt:
            system = None
        else:
            system = system_prompt.format(
                format=format_schema,
                n_clusters_instruction=n_clusters_instruction,
                validation_rules=validation_rules,
            )

        # Build user prompt (or None if falsy)
        if not user_prompt:
            user = None
        else:
            user = user_prompt.format(items=items_str)

        return system, user

    def _validate_clusters(
        self,
        response: BaseModel,
        n_items: int,
        allow_overlap: bool,
        require_all: bool,
    ) -> list[str]:
        """Validate the clustering response for referential integrity.

        Args:
            response: The parsed LLM response
            n_items: Total number of input items
            allow_overlap: Whether items can appear in multiple clusters
            require_all: Whether all items must be assigned

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        valid_ids = set(range(1, n_items + 1))
        seen_ids: set[int] = set()

        clusters = getattr(response, "clusters", [])

        for i, cluster in enumerate(clusters):
            reference_ids = getattr(cluster, "reference_ids", [])

            # Check for empty clusters
            if not reference_ids:
                errors.append(f"Cluster {i + 1} has no references assigned")
                continue

            for reference_id in reference_ids:
                # Check for invalid IDs
                if reference_id not in valid_ids:
                    errors.append(
                        f"Invalid reference ID {reference_id} in cluster {i + 1} "
                        f"(valid range: 1 to {n_items})"
                    )
                # Check for duplicates across clusters
                elif not allow_overlap and reference_id in seen_ids:
                    errors.append(
                        f"Reference ID {reference_id} appears in multiple clusters "
                        f"(overlap not allowed)"
                    )

                seen_ids.add(reference_id)

        # Check for missing IDs
        if require_all:
            missing_ids = valid_ids - seen_ids
            if missing_ids:
                missing_str = ", ".join(str(x) for x in sorted(missing_ids))
                errors.append(f"Reference IDs not assigned to any cluster: {missing_str}")

        return errors

    def _is_context_length_error(self, error: Exception) -> bool:
        """Check if an exception is a context length error."""
        error_str = str(error).lower()
        context_indicators = [
            "context length",
            "context_length",
            "token limit",
            "max_tokens",
            "maximum context",
            "too long",
            "too many tokens",
            "exceeds the model's maximum",
            "maximum token",
        ]
        return any(indicator in error_str for indicator in context_indicators)

    def cluster(
        self,
        inputs: list[tuple[int, str]],
        cluster_schema: Type[T],
        n_clusters: int | str | None = None,
        allow_overlap: bool = False,
        require_all: bool = True,
        max_retries: int = 3,
        validation_retries: int = 2,
        system_prompt: str | None = DEFAULT_CLUSTER_SYSTEM_PROMPT,
        user_prompt: str | None = DEFAULT_CLUSTER_USER_PROMPT,
        only_system_prompt: bool = False,
        validate_placeholders: bool = True,
        **llm_kwargs: Any,
    ) -> ClusterResult[T]:
        """Cluster multiple items using a single LLM call.

        Args:
            inputs: List of tuples (index, text) to cluster
            cluster_schema: Pydantic model defining per-cluster fields (e.g., name, summary).
                A 'reference_ids' field will be automatically added to track item assignments.
            n_clusters: Optional hint for the number of clusters to create.
                If None, the LLM decides based on natural groupings.
            allow_overlap: If True, items can appear in multiple clusters.
                If False (default), each item must be in exactly one cluster.
            require_all: If True (default), every item must be assigned to a cluster.
                If False, items may be left unassigned.
            max_retries: Number of retries for Instructor on structural failures
                (malformed JSON, wrong types). Default 3.
            validation_retries: Number of retries on post-hoc validation failures
                (invalid IDs, duplicates, missing items). Default 2.
            system_prompt: Override system prompt (None->exclude)
            user_prompt: Override user prompt (None->exclude)
            only_system_prompt: If True, combine system and user prompts into one
            **llm_kwargs: Additional kwargs passed to the LLM (e.g., temperature)

        Returns:
            ClusterResult containing the clusters with resolved items

        Raises:
            ValueError: If inputs is empty or n_clusters < 1
            ClusterValidationError: If validation fails after all retries
            ContextLengthError: If input exceeds model's context window
        """
        # Input validation
        if not inputs:
            raise ValueError("inputs cannot be empty")

        # Validate unique indices and inputs structure
        indices = set()
        for i, item in inputs:
            if not isinstance(i, int) or not isinstance(item, str):
                raise ValueError("Each input must be a tuple of (int, str)")
            if i in indices:
                raise ValueError(f"Duplicate index {i} in inputs")
            indices.add(i)

        if n_clusters is not None and (isinstance(n_clusters, int) and n_clusters < 1):
            raise ValueError("n_clusters must be at least 1")

        n_items = len(inputs)

        # Build the response schema
        response_schema = self._build_cluster_schema(
            cluster_schema, n_items
        )

        # Build initial prompts
        system, user = self._build_prompt(
            inputs, response_schema, n_clusters, allow_overlap, require_all,
            system_prompt, user_prompt, validate_placeholders
        )

        # Build messages list
        messages: list[dict[str, str]] = []
        if only_system_prompt:
            # Combine system and user into a single system message
            parts = [p for p in [system, user] if p is not None]
            combined = "\n\n".join(parts)
            if combined:
                messages.append({"role": "system", "content": combined})
        else:
            if system is not None:
                messages.append({"role": "system", "content": system})
            if user is not None:
                messages.append({"role": "user", "content": user})

        if not messages:
            raise ValueError(
                "No prompt messages were generated. Ensure at least one of "
                "system_prompt or user_prompt is non-empty."
            )

        retries_used = 0
        last_errors: list[str] = []

        for attempt in range(validation_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    messages=messages,
                    response_model=response_schema,
                    max_retries=max_retries,
                    **llm_kwargs,
                )
            except Exception as e:
                if self._is_context_length_error(e):
                    raise ContextLengthError(e) from e
                raise

            # Validate the response
            validation_errors = self._validate_clusters(
                response, n_items, allow_overlap, require_all
            )

            if not validation_errors:
                # Success - build the result
                return self._build_result(
                    response, inputs, cluster_schema, retries_used
                )

            # Validation failed
            last_errors = validation_errors
            retries_used += 1

            if attempt < validation_retries:
                # Add correction message and retry
                correction_msg = (
                    "Your previous response had the following issues:\n"
                    + "\n".join(f"- {e}" for e in validation_errors)
                    + "\n\nPlease fix these issues and return the corrected clustering."
                )
                messages.append({"role": "user", "content": correction_msg})

        # All retries exhausted
        raise ClusterValidationError(
            last_errors,
            f"Cluster validation failed after {validation_retries + 1} attempts: "
            + "; ".join(last_errors),
        )

    def _build_result(
        self,
        response: BaseModel,
        inputs: list[tuple[int, str]],
        cluster_schema: Type[T],
        retries_used: int,
    ) -> ClusterResult[T]:
        """Build the ClusterResult from a validated response."""
        clusters_data = getattr(response, "clusters", [])

        cluster_items: list[ClusterItem[T]] = []

        for cluster_data in clusters_data:
            # Extract reference_ids
            reference_ids = getattr(cluster_data, "reference_ids", [])

            # Resolve items from IDs (1-indexed)
            references = [inputs[i - 1] for i in reference_ids]

            # Reconstruct the user's cluster schema (without reference_ids)
            cluster_dict = {}
            for field_name in cluster_schema.model_fields:
                if field_name != "reference_ids":
                    cluster_dict[field_name] = getattr(cluster_data, field_name)

            cluster_instance = cluster_schema(**cluster_dict)

            cluster_items.append(ClusterItem(
                cluster=cluster_instance,
                references=references,
            ))

        return ClusterResult(
            clusters=cluster_items,
            raw_response=response,
            retries_used=retries_used,
        )
