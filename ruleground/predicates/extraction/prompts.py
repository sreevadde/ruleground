"""
Extraction Prompts for Weak Supervision

Prompt templates for extracting predicate labels from human rationales
using LLMs (Paper Section 5.1, Stage 0).
"""

from __future__ import annotations

import json
from typing import Dict, List

from ruleground.predicates.ontology import (
    PREDICATE_ONTOLOGY,
    Sport,
    SPORT_FROM_STR,
    get_predicates_for_sport,
)


def _build_ontology_json(sport: str | None = None) -> str:
    """Build a JSON description of predicates for the prompt."""
    if sport:
        preds = get_predicates_for_sport(sport)
    else:
        preds = PREDICATE_ONTOLOGY

    entries = []
    for p in preds:
        entries.append({
            "name": p.name,
            "type": p.ptype.name.lower(),
            "sport": p.sport.name.lower(),
            "description": p.description,
        })
    return json.dumps(entries, indent=2)


SYSTEM_PROMPT = """You are a sports rule predicate extractor. You analyze human-written \
rationales explaining referee decisions and extract structured predicate labels.

You must:
1. Extract ONLY predicates that are explicitly stated or logically implied by the text.
2. Do NOT infer predicates without textual support.
3. Assign confidence based on explicitness:
   - 1.0 = directly stated
   - 0.8-0.9 = strongly implied
   - 0.6-0.7 = weakly implied
4. Return valid JSON only."""


EXTRACTION_TEMPLATE = """## Predicate Ontology
{ontology_json}

## Task
Given the following rationale explaining a referee's decision, extract all rule-relevant \
predicates with their boolean values and confidence scores.

## Sport
{sport}

## Rationale
{rationale}

## Required Output Format (JSON)
{{
  "predicates": [
    {{"name": "<predicate_name>", "value": true, "confidence": 0.95}},
    {{"name": "<predicate_name>", "value": false, "confidence": 0.8}}
  ],
  "reasoning": "<brief explanation of extraction logic>"
}}"""


def build_extraction_prompt(
    rationale: str,
    sport: str,
    include_ontology: bool = True,
) -> str:
    """Build a full extraction prompt for a single rationale.

    Args:
        rationale:       Human-written explanation of a referee decision.
        sport:           Sport name (basketball, football, soccer).
        include_ontology: Whether to include the predicate ontology.

    Returns:
        Formatted prompt string.
    """
    ontology = _build_ontology_json(sport) if include_ontology else "See system prompt."
    return EXTRACTION_TEMPLATE.format(
        ontology_json=ontology,
        sport=sport,
        rationale=rationale,
    )


VALIDATION_TEMPLATE = """## Task
You are validating predicate extractions from a sports rationale.
Compare the extracted predicates against the original rationale and assess accuracy.

## Rationale
{rationale}

## Extracted Predicates
{extracted_json}

## Instructions
For each extracted predicate, judge:
1. Is the extraction correct? (agree/disagree)
2. Are any predicates missing?
3. Are confidence scores reasonable?

## Output (JSON)
{{
  "validations": [
    {{"name": "<predicate_name>", "agree": true, "corrected_value": null, "corrected_confidence": null}},
    {{"name": "<predicate_name>", "agree": false, "corrected_value": false, "corrected_confidence": 0.6}}
  ],
  "missing_predicates": [
    {{"name": "<predicate_name>", "value": true, "confidence": 0.85}}
  ]
}}"""


def build_validation_prompt(
    rationale: str,
    extracted: List[Dict],
) -> str:
    """Build a validation prompt for cross-model agreement."""
    return VALIDATION_TEMPLATE.format(
        rationale=rationale,
        extracted_json=json.dumps(extracted, indent=2),
    )
