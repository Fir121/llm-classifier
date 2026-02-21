"""Core LLMClassifier implementation."""

import hashlib
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Type, TypeVar

import instructor
from pydantic import BaseModel, Field, create_model

T = TypeVar("T", bound=BaseModel)


# ============================================================================
# Result Types
# ============================================================================


@dataclass
class PredictResult(Generic[T]):
    """Result of a single prediction.

    Attributes:
        value: The predicted output matching the user's schema
        reasoning: LLM's reasoning for its prediction (if reasoning=True)
        confidence: LLM's confidence score 0-1 (if confidence=True)
        compliant_variants: Consensus-compliant outputs (when consensus > 1)
        noncompliant_variants: Consensus-rejected outputs (when consensus > 1)
        raw_response: The raw wrapped response from the LLM
    """

    value: T
    reasoning: str | None = None
    confidence: float | None = None
    compliant_variants: list[T] | None = None
    noncompliant_variants: list[Any] | None = None
    raw_response: BaseModel | None = None


@dataclass
class BatchResult(Generic[T]):
    """Result of a batch prediction.

    Attributes:
        results: List of PredictResult for each input
        successes: Number of successful predictions
        failures: Number of failed predictions
        errors: List of (index, error) tuples for failed predictions
    """

    results: list[PredictResult[T] | None]
    successes: int = 0
    failures: int = 0
    errors: list[tuple[int, Exception]] = field(default_factory=list)

    def values(self) -> list[T | None]:
        """Get just the predicted values, None for failures."""
        return [r.value if r else None for r in self.results]


# ============================================================================
# Wrapper Utilities
# ============================================================================


def create_wrapped_model(
    base_schema: Type[BaseModel],
    include_reasoning: bool = False,
    include_confidence: bool = False,
) -> Type[BaseModel]:
    """Dynamically create a Pydantic model that adds reasoning/confidence to the user's schema.

    When reasoning=True, adds a 'reasoning' field at the start that the LLM fills with
    its step-by-step thought process before producing the result.

    When confidence=True, adds a 'confidence' field at the end (0-1) representing
    the LLM's self-assessed confidence in its prediction.

    Args:
        base_schema: The user's Pydantic model class
        include_reasoning: Whether to add a reasoning field
        include_confidence: Whether to add a confidence field

    Returns:
        A new Pydantic model class with the additional fields merged in

    Raises:
        ValueError: If the base schema already has 'reasoning' or 'confidence' fields
    """
    if not include_reasoning and not include_confidence:
        return base_schema

    # Check for conflicting field names
    base_fields = base_schema.model_fields.keys()
    if include_reasoning and "reasoning" in base_fields:
        raise ValueError("Base schema already has a 'reasoning' field")
    if include_confidence and "confidence" in base_fields:
        raise ValueError("Base schema already has a 'confidence' field")

    # Build fields in order: reasoning, base fields, confidence
    extra_fields: dict = {}

    if include_reasoning:
        extra_fields["reasoning"] = (
            str,
            Field(
                description=(
                    "Your step-by-step reasoning process for arriving at this prediction. "
                    "Explain your thought process clearly and concisely."
                )
            ),
        )

    # Add all base schema fields
    for field_name, field_info in base_schema.model_fields.items():
        extra_fields[field_name] = (field_info.annotation, field_info)

    if include_confidence:
        extra_fields["confidence"] = (
            float,
            Field(
                ge=0.0,
                le=1.0,
                description=(
                    "Your confidence in this prediction, from 0.0 (no confidence) to 1.0 "
                    "(completely certain). Be calibrated - use lower values when uncertain."
                ),
            ),
        )

    # Create the wrapped model
    wrapped_model = create_model(base_schema.__name__, **extra_fields)

    return wrapped_model


def unwrap_response(
    response: BaseModel,
    base_schema: Type[BaseModel],
    include_reasoning: bool = False,
    include_confidence: bool = False,
) -> tuple[BaseModel, str | None, float | None]:
    """Extract the result and optional fields from a wrapped response.

    Args:
        response: The wrapped model instance from the LLM
        base_schema: The user's original Pydantic model class
        include_reasoning: Whether reasoning was requested
        include_confidence: Whether confidence was requested

    Returns:
        Tuple of (result, reasoning, confidence)
    """
    if not include_reasoning and not include_confidence:
        # No wrapping was done, response is the raw result
        return response, None, None

    # Extract reasoning and confidence if present
    reasoning = getattr(response, "reasoning", None) if include_reasoning else None
    confidence = getattr(response, "confidence", None) if include_confidence else None

    # Reconstruct the base model from all fields except reasoning and confidence
    result_data = {
        field_name: getattr(response, field_name)
        for field_name in base_schema.model_fields.keys()
    }
    result = base_schema(**result_data)

    return result, reasoning, confidence


# ============================================================================
# Prompts
# ============================================================================

DEFAULT_SYSTEM_PROMPT = """Analyze the input and respond with a JSON object.
Output schema:
{format}{examples}"""

DEFAULT_USER_PROMPT = "{input}"


class LLMClassifier:
    """LLM-based classifier/extractor with structured outputs.

    Get structured Pydantic model outputs from LLMs with automatic
    validation and retries for all zero/few shot classification tasks.

    Args:
        model: Model identifier in format "provider/model-name"
            Examples: "openai/gpt-4.1", "anthropic/claude-4-6-sonnet"
        api_key: Optional API key (defaults to environment variables)
        **client_kwargs: Additional kwargs passed to the Instructor client

    Example:
        >>> from pydantic import BaseModel
        >>> from typing import Literal
        >>>
        >>> class Sentiment(BaseModel):
        ...     label: Literal["positive", "negative", "neutral"]
        >>>
        >>> clf = LLMClassifier(model="openai/gpt-4.1")
        >>> result = clf.predict("I love this!", output_schema=Sentiment)
        >>> print(result.value.label)
        'positive'
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

    def _format_examples(
        self,
        examples: list[tuple[str, BaseModel]] | None,
        reasoning: bool = False,
        confidence: bool = False,
    ) -> str:
        """Format examples into a string for the prompt.

        If reasoning or confidence flags are enabled, wraps each example output
        with placeholder values to demonstrate the expected output structure.
        """
        if not examples:
            return ""

        formatted = []
        for i, (input_text, output) in enumerate(examples, 1):
            # Build output dict with reasoning/confidence wrappers if enabled
            output_dict = {}

            if reasoning and not hasattr(output, "reasoning"):
                output_dict["reasoning"] = "<your step-by-step reasoning>"

            # Add all fields from the example output
            output_dict.update(output.model_dump())

            if confidence and not hasattr(output, "confidence"):
                output_dict["confidence"] = "<your confidence score>"

            output_json = json.dumps(output_dict, indent=2)
            formatted.append(f"{i}. Input: \"{input_text}\"\n   Output: {output_json}")

        examples_str = "\n\n".join(formatted)
        return f"\n\nExamples:\n{examples_str}"

    def _build_prompt(
        self,
        input_text: str,
        output_schema: Type[BaseModel],
        examples: list[tuple[str, BaseModel]] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        reasoning: bool = False,
        confidence: bool = False,
        validate_placeholders: bool = True,
    ) -> tuple[str | None, str | None]:
        """Build the system and user prompts from templates and components."""
        examples_str = self._format_examples(examples, reasoning, confidence)
        format_schema = json.dumps(output_schema.model_json_schema(), indent=2)

        # Raise an error if any placeholder has a value but appears in neither prompt.
        if validate_placeholders:
            params = [
                ("{examples}", examples_str),
                ("{format}", format_schema),
                ("{input}", input_text),
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


        # Build system prompt (or None if empty string)
        if not system_prompt:
            system = None
        else:
            system = system_prompt.format(
                examples=examples_str,
                format=format_schema,
                input=input_text,
            )

        # Build user prompt (or None if empty string)
        if not user_prompt:
            user = None
        else:
            user = user_prompt.format(
                examples=examples_str,
                format=format_schema,
                input=input_text,
            )

        return system, user

    def _single_predict(
        self,
        input_text: str,
        wrapped_schema: Type[BaseModel],
        examples: list[tuple[str, BaseModel]] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_retries: int = 3,
        reasoning: bool = False,
        confidence: bool = False,
        only_system_prompt: bool = False,
        validate_placeholders: bool = True,
        **llm_kwargs: Any,
    ) -> BaseModel:
        """Make a single LLM prediction call."""
        system, user = self._build_prompt(
            input_text, wrapped_schema, examples, system_prompt, user_prompt,
            reasoning=reasoning, confidence=confidence,
            validate_placeholders=validate_placeholders,
        )

        # Build messages list, excluding None prompts
        messages = []
        if only_system_prompt:
            # Combine system and user into a single system message
            parts = [p for p in [system, user] if p is not None]
            combined_prompt = "\n\n".join(parts)
            if combined_prompt:
                messages.append({"role": "system", "content": combined_prompt})
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

        response = self._client.chat.completions.create(
            messages=messages,
            response_model=wrapped_schema,
            max_retries=max_retries,
            **llm_kwargs,
        )

        return response

    def _to_cache_safe(self, value: Any) -> Any:
        """Convert values into deterministic JSON-serializable forms for caching."""
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {str(k): self._to_cache_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_cache_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _build_input_cache_key(
        self,
        input_text: str,
        output_schema: Type[BaseModel],
        examples: list[tuple[str, BaseModel]] | None,
        reasoning: bool,
        confidence: bool,
        consensus: int,
        system_prompt: str | None,
        user_prompt: str | None,
        only_system_prompt: bool,
        llm_kwargs: dict[str, Any],
    ) -> str:
        """Build a deterministic cache key for a single input.

        The key is based on all factors that affect the output:
        model, input text, schema, examples, prompts, and prediction settings.
        """
        payload = {
            "model": self.model,
            "input": input_text,
            "output_schema_name": output_schema.__name__,
            "output_schema": output_schema.model_json_schema(),
            "examples": [
                {
                    "input": ex_input,
                    "output": ex_output.model_dump(mode="json"),
                }
                for ex_input, ex_output in (examples or [])
            ],
            "reasoning": reasoning,
            "confidence": confidence,
            "consensus": consensus,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "only_system_prompt": only_system_prompt,
            "llm_kwargs": llm_kwargs,
        }
        normalized = json.dumps(
            self._to_cache_safe(payload), sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _serialize_predict_result_for_cache(self, result: PredictResult[T]) -> dict[str, Any]:
        """Serialize PredictResult to a JSON-safe dictionary."""
        return {
            "value": result.value.model_dump(mode="json"),
            "reasoning": result.reasoning,
            "confidence": result.confidence,
            "compliant_variants": [
                variant.model_dump(mode="json") for variant in result.compliant_variants
            ] if result.compliant_variants is not None else None,
            "noncompliant_variants": [
                self._to_cache_safe(variant) for variant in result.noncompliant_variants
            ] if result.noncompliant_variants is not None else None,
        }

    def _deserialize_predict_result_from_cache(
        self,
        data: dict[str, Any],
        output_schema: Type[T],
    ) -> PredictResult[T]:
        """Reconstruct PredictResult from cached serialized form."""
        compliant_raw = data.get("compliant_variants")
        compliant_variants = [
            output_schema.model_validate(item) for item in compliant_raw
        ] if compliant_raw is not None else None

        noncompliant_raw = data.get("noncompliant_variants")
        noncompliant_variants = [
            output_schema.model_validate(item) for item in noncompliant_raw
        ] if noncompliant_raw is not None else None

        return PredictResult(
            value=output_schema.model_validate(data["value"]),
            reasoning=data.get("reasoning"),
            confidence=data.get("confidence"),
            compliant_variants=compliant_variants,
            noncompliant_variants=noncompliant_variants,
            raw_response=None,
        )

    def _append_input_cache_record(
        self,
        records_path: Path,
        cache_key: str,
        result: PredictResult[T],
        lock: Any = None,
    ) -> None:
        """Append a successful input result to the cache file."""
        record: dict[str, Any] = {
            "key": cache_key,
            "result": self._serialize_predict_result_for_cache(result),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"

        if lock is not None:
            with lock:
                with records_path.open("a", encoding="utf-8") as file:
                    file.write(line)
        else:
            with records_path.open("a", encoding="utf-8") as file:
                file.write(line)

    def _load_input_cache(
        self,
        records_path: Path,
        output_schema: Type[T],
    ) -> dict[str, PredictResult[T]]:
        """Load cached results keyed by input cache key."""
        if not records_path.exists():
            return {}

        cache: dict[str, PredictResult[T]] = {}
        with records_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record.get("key")
                if key and "result" in record:
                    cache[key] = self._deserialize_predict_result_from_cache(
                        record["result"], output_schema
                    )
        return cache

    def predict(
        self,
        input: str,
        output_schema: Type[T],
        examples: list[tuple[str, BaseModel]] | None = None,
        reasoning: bool = False,
        confidence: bool = False,
        consensus: int = 1,
        consensus_parallel: bool = False,
        max_parallel: int = 5,
        max_retries: int = 3,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str | None = DEFAULT_USER_PROMPT,
        only_system_prompt: bool = False,
        validate_placeholders: bool = True,
        **llm_kwargs: Any,
    ) -> PredictResult[T]:
        """Predict structured output for a single input.

        Args:
            input: The text input to classify/predict on
            output_schema: Pydantic model class defining the output structure
            examples: Optional list of (input, output) tuples for few-shot learning
            reasoning: If True, ask LLM to explain its reasoning
            confidence: If True, ask LLM to provide a confidence score (0-1)
            consensus: Number of times to run prediction for majority voting (default: 1)
            consensus_parallel: Whether to run consensus calls in parallel (default: False)
            max_parallel: Maximum number of parallel consensus calls (default: 5)
            max_retries: Maximum retries per LLM call on validation failure (default: 3)
            system_prompt: Override system prompt (None->exclude)
            user_prompt: Override user prompt (None->exclude)
            only_system_prompt: If True, combine system and user prompts into one
            validate_placeholders: If True, error if placeholders missing from prompts
            **llm_kwargs: Additional kwargs passed to the LLM (temperature, max_tokens, etc.)

        Returns:
            PredictResult containing the prediction, optional reasoning/confidence, and variants
        """
        if consensus < 1:
            raise ValueError("consensus must be >= 1")
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")

        # Create wrapped schema if needed
        wrapped_schema = create_wrapped_model(
            output_schema,
            include_reasoning=reasoning,
            include_confidence=confidence,
        )

        if consensus <= 1:
            # Single prediction
            response = self._single_predict(
                input_text=input,
                wrapped_schema=wrapped_schema,
                examples=examples,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_retries=max_retries,
                reasoning=reasoning,
                confidence=confidence,
                only_system_prompt=only_system_prompt,
                validate_placeholders=validate_placeholders,
                **llm_kwargs,
            )

            value, reasoning_text, confidence_score = unwrap_response(
                response, output_schema, reasoning, confidence
            )

            return PredictResult(
                value=value,
                reasoning=reasoning_text,
                confidence=confidence_score,
                compliant_variants=None,
                noncompliant_variants=None,
                raw_response=response,
            )

        # Consensus mode: run multiple predictions
        responses: list[BaseModel] = []

        if consensus_parallel:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=min(consensus, max_parallel)) as executor:
                futures = [
                    executor.submit(
                        self._single_predict,
                        input_text=input,
                        wrapped_schema=wrapped_schema,
                        examples=examples,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_retries=max_retries,
                        reasoning=reasoning,
                        confidence=confidence,
                        only_system_prompt=only_system_prompt,
                        validate_placeholders=validate_placeholders,
                        **llm_kwargs,
                    )
                    for _ in range(consensus)
                ]
                responses = [f.result() for f in futures]
        else:
            # Sequential execution
            for _ in range(consensus):
                response = self._single_predict(
                    input_text=input,
                    wrapped_schema=wrapped_schema,
                    examples=examples,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_retries=max_retries,
                    reasoning=reasoning,
                    confidence=confidence,
                    only_system_prompt=only_system_prompt,
                    validate_placeholders=validate_placeholders,
                    **llm_kwargs,
                )
                responses.append(response)

        # Unwrap all responses
        unwrapped: list[tuple[BaseModel, str | None, float | None]] = [
            unwrap_response(r, output_schema, reasoning, confidence) for r in responses
        ]

        # Extract just the values for voting
        values = [u[0] for u in unwrapped]

        # Find majority by JSON serialization (handles complex objects)
        value_jsons = [v.model_dump_json() for v in values]
        counter = Counter(value_jsons)
        majority_json = counter.most_common(1)[0][0]

        # Find the first response that matches the majority
        majority_idx = value_jsons.index(majority_json)
        majority_value, majority_reasoning, majority_confidence = unwrapped[majority_idx]

        # Build compliant and noncompliant variants
        compliant = [
            u[0] for u in unwrapped
            if u[0].model_dump_json() == majority_json
        ] if consensus > 1 else None
        noncompliant = [
            u[0] for u in unwrapped
            if u[0].model_dump_json() != majority_json
        ] if consensus > 1 else None

        return PredictResult(
            value=majority_value,
            reasoning=majority_reasoning,
            confidence=majority_confidence,
            compliant_variants=compliant,
            noncompliant_variants=noncompliant,
            raw_response=responses[majority_idx],
        )

    def batch_predict(
        self,
        inputs: list[str],
        output_schema: Type[T],
        examples: list[tuple[str, BaseModel]] | None = None,
        reasoning: bool = False,
        confidence: bool = False,
        consensus: int = 1,
        consensus_parallel: bool = True,
        parallel: bool = True,
        max_parallel: int = 5,
        max_retries: int = 3,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str | None = DEFAULT_USER_PROMPT,
        only_system_prompt: bool = False,
        validate_placeholders: bool = True,
        cache_dir: str | None = None,
        cache_key: str | None = None,
        **llm_kwargs: Any,
    ) -> BatchResult[T]:
        """Predict structured output for multiple inputs.

        Args:
            inputs: List of text inputs to classify/predict on
            output_schema: Pydantic model class defining the output structure
            examples: Optional list of (input, output) tuples for few-shot learning
            reasoning: If True, ask LLM to explain its reasoning
            confidence: If True, ask LLM to provide a confidence score (0-1)
            consensus: Number of times to run each prediction for majority voting
            consensus_parallel: Whether to run consensus calls in parallel
            parallel: Whether to process inputs in parallel (default: True)
            max_parallel: Maximum number of parallel input predictions (default: 5)
            max_retries: Maximum retries per LLM call on validation failure (default: 3)
            system_prompt: Override system prompt (None->exclude)
            user_prompt: Override user prompt (None->exclude)
            only_system_prompt: If True, combine system and user prompts into one
            validate_placeholders: If True, error if placeholders missing from prompts
            cache_dir: Optional directory for per-input cache files. Results are cached
                based on input text, examples, prompts, consensus, reasoning, and confidence.
            cache_key: Optional cache file name (without extension). Defaults to "input_cache".
            **llm_kwargs: Additional kwargs passed to the LLM

        Returns:
            BatchResult containing all predictions, success/failure counts, and errors
        """
        if not inputs:
            raise ValueError("inputs must not be empty")
        if consensus < 1:
            raise ValueError("consensus must be >= 1")
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        if cache_key is not None and cache_dir is None:
            raise ValueError("cache_key requires cache_dir")

        results: list[PredictResult[T] | None] = [None] * len(inputs)
        errors_map: dict[int, Exception] = {}
        cache_lock = threading.Lock()

        records_path: Path | None = None
        input_cache: dict[str, PredictResult[T]] = {}
        input_cache_keys: list[str] = []

        # Pre-compute cache keys for all inputs
        for input_text in inputs:
            input_cache_keys.append(
                self._build_input_cache_key(
                    input_text=input_text,
                    output_schema=output_schema,
                    examples=examples,
                    reasoning=reasoning,
                    confidence=confidence,
                    consensus=consensus,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    only_system_prompt=only_system_prompt,
                    llm_kwargs=llm_kwargs,
                )
            )

        if cache_dir is not None:
            cache_root = Path(cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            resolved_key = cache_key if cache_key else "input_cache"
            records_path = cache_root / f"{resolved_key}.jsonl"

            # Load existing cache
            input_cache = self._load_input_cache(
                records_path=records_path,
                output_schema=output_schema,
            )

            # Check cache for each input
            for idx, key in enumerate(input_cache_keys):
                if key in input_cache:
                    results[idx] = input_cache[key]

        def predict_single(
            idx: int, input_text: str, cache_key: str
        ) -> tuple[int, PredictResult[T] | None, Exception | None, str]:
            try:
                result = self.predict(
                    input=input_text,
                    output_schema=output_schema,
                    examples=examples,
                    reasoning=reasoning,
                    confidence=confidence,
                    consensus=consensus,
                    consensus_parallel=consensus_parallel,
                    max_parallel=max_parallel,
                    max_retries=max_retries,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    only_system_prompt=only_system_prompt,
                    validate_placeholders=validate_placeholders,
                    **llm_kwargs,
                )
                return idx, result, None, cache_key
            except Exception as e:
                return idx, None, e, cache_key

        pending_items = [
            (idx, input_text, input_cache_keys[idx])
            for idx, input_text in enumerate(inputs)
            if results[idx] is None
        ]

        if parallel and pending_items:
            # Parallel batch processing
            with ThreadPoolExecutor(max_workers=min(len(pending_items), max_parallel)) as executor:
                futures = [
                    executor.submit(predict_single, idx, input_text, cache_key)
                    for idx, input_text, cache_key in pending_items
                ]

                for future in futures:
                    idx, result, error, cache_key = future.result()
                    if error:
                        errors_map[idx] = error
                    else:
                        errors_map.pop(idx, None)
                        results[idx] = result
                        if records_path is not None and result is not None:
                            self._append_input_cache_record(
                                records_path, cache_key, result, cache_lock
                            )
        else:
            # Sequential batch processing
            for idx, input_text, cache_key in pending_items:
                _, result, error, _ = predict_single(idx, input_text, cache_key)
                if error:
                    errors_map[idx] = error
                else:
                    errors_map.pop(idx, None)
                    results[idx] = result
                    if records_path is not None and result is not None:
                        self._append_input_cache_record(records_path, cache_key, result)

        successes = sum(result is not None for result in results)
        failures = len(results) - successes
        errors = [(idx, error) for idx, error in sorted(errors_map.items())]

        return BatchResult(
            results=results,
            successes=successes,
            failures=failures,
            errors=errors,
        )
