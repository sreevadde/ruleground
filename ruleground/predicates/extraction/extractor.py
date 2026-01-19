"""
LLM-based Predicate Extractor

Extracts predicate labels from human rationales using Claude or GPT-4o
(Paper Section 5.1, Stage 0: Weak Supervision).

88.3% cross-model agreement (Claude <-> GPT-4o), Cohen's kappa = 0.76.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ruleground.predicates.ontology import (
    ALL_PREDICATE_NAMES,
    PREDICATE_NAME_TO_IDX,
    get_predicate_names_for_sport,
)
from ruleground.predicates.extraction.prompts import (
    SYSTEM_PROMPT,
    build_extraction_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class PredicateExtraction:
    """Result of extracting predicates from a single rationale."""

    video_id: str
    sport: str
    predicates: Dict[str, bool] = field(default_factory=dict)
    confidences: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    raw_response: str = ""

    def to_label_dict(self) -> Dict[str, Any]:
        """Convert to training label format."""
        labels = {}
        weights = {}
        for name in ALL_PREDICATE_NAMES:
            if name in self.predicates:
                labels[name] = float(self.predicates[name])
                weights[name] = self.confidences.get(name, 0.5)
            else:
                labels[name] = 0.0
                weights[name] = 0.0  # mask: unextracted predicates
        return {"labels": labels, "weights": weights}


class PredicateExtractor:
    """Extract predicate labels from rationales using LLM APIs.

    Supports both Anthropic (Claude) and OpenAI (GPT-4o) backends.

    Args:
        backend:      'anthropic' or 'openai'.
        model:        Model identifier.
        temperature:  Sampling temperature (low for extraction).
        max_retries:  Number of retries on parse failure.
    """

    def __init__(
        self,
        backend: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.1,
        max_retries: int = 3,
    ):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        """Lazily initialize API client."""
        if self._client is None:
            if self.backend == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic()
            elif self.backend == "openai":
                import openai
                self._client = openai.OpenAI()
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
        return self._client

    def _call_anthropic(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _call_openai(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def _call_llm(self, prompt: str) -> str:
        if self.backend == "anthropic":
            return self._call_anthropic(prompt)
        return self._call_openai(prompt)

    def _parse_response(
        self, raw: str, sport: str
    ) -> tuple[Dict[str, bool], Dict[str, float], str]:
        """Parse LLM response into structured predicates."""
        # Extract JSON from response (handle markdown code blocks)
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        data = json.loads(text.strip())
        valid_names = set(get_predicate_names_for_sport(sport))

        predicates = {}
        confidences = {}
        for entry in data.get("predicates", []):
            name = entry["name"]
            if name not in valid_names:
                logger.warning(f"Ignoring unknown predicate: {name}")
                continue
            predicates[name] = bool(entry["value"])
            confidences[name] = float(entry.get("confidence", 0.5))

        reasoning = data.get("reasoning", "")
        return predicates, confidences, reasoning

    def extract(
        self,
        rationale: str,
        sport: str,
        video_id: str = "",
    ) -> PredicateExtraction:
        """Extract predicates from a single rationale.

        Args:
            rationale: Human-written explanation of a referee decision.
            sport:     Sport name.
            video_id:  Optional video identifier.

        Returns:
            PredicateExtraction with parsed predicates and confidences.
        """
        prompt = build_extraction_prompt(rationale, sport)

        for attempt in range(self.max_retries):
            try:
                raw = self._call_llm(prompt)
                predicates, confidences, reasoning = self._parse_response(raw, sport)
                return PredicateExtraction(
                    video_id=video_id,
                    sport=sport,
                    predicates=predicates,
                    confidences=confidences,
                    reasoning=reasoning,
                    raw_response=raw,
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Parse failure (attempt {attempt + 1}): {e}")
                continue

        logger.error(f"All {self.max_retries} attempts failed for {video_id}")
        return PredicateExtraction(video_id=video_id, sport=sport)

    def extract_batch(
        self,
        items: List[Dict[str, str]],
    ) -> List[PredicateExtraction]:
        """Extract predicates from a batch of rationales.

        Args:
            items: List of dicts with 'rationale', 'sport', and optionally 'video_id'.

        Returns:
            List of PredicateExtraction results.
        """
        results = []
        for item in items:
            result = self.extract(
                rationale=item["rationale"],
                sport=item["sport"],
                video_id=item.get("video_id", ""),
            )
            results.append(result)
        return results
