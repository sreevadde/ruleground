"""
Extraction CLI

Extract predicates from human rationales using LLM APIs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from ruleground.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def run_extraction(
    data_path: str,
    output_path: str,
    backend: str = "anthropic",
    model_name: str = "claude-sonnet-4-20250514",
    validate: bool = False,
) -> None:
    """Extract predicates from rationales.

    Args:
        data_path:   Path to JSON file with rationales.
        output_path: Where to save extracted predicates.
        backend:     LLM backend ('anthropic' or 'openai').
        model_name:  Model identifier.
        validate:    Whether to run cross-model validation.
    """
    setup_logging()

    # Load data
    with open(data_path) as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} rationales from {data_path}")

    from ruleground.predicates.extraction.extractor import PredicateExtractor

    extractor = PredicateExtractor(
        backend=backend,
        model=model_name,
    )

    # Extract
    results = extractor.extract_batch(data)

    # Build output: video_id -> {labels, weights}
    output = {}
    for r in results:
        if r.video_id:
            output[r.video_id] = r.to_label_dict()

    logger.info(f"Extracted predicates for {len(output)} videos")

    # Validate if requested
    if validate:
        logger.info("Running cross-model validation...")
        from ruleground.predicates.extraction.validator import CrossModelValidator

        secondary_backend = "openai" if backend == "anthropic" else "anthropic"
        secondary_model = "gpt-4o" if secondary_backend == "openai" else "claude-sonnet-4-20250514"

        secondary = PredicateExtractor(backend=secondary_backend, model=secondary_model)
        validator = CrossModelValidator(extractor, secondary)
        val_results, stats = validator.validate_batch(data)

        logger.info(f"Cross-model agreement: {stats['mean_agreement']:.1%}")
        logger.info(f"Cohen's kappa: {stats.get('cohens_kappa', 'N/A')}")

        # Use merged results
        for r in val_results:
            if r.video_id:
                output[r.video_id] = {
                    "labels": {k: float(v) for k, v in r.merged_predicates.items()},
                    "weights": r.merged_confidences,
                }

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved predicates to {output_path}")


def main():
    """Entry point for rg-extract command."""
    import sys
    if len(sys.argv) < 5:
        print("Usage: rg-extract -d <data.json> -o <output.json> [-b backend] [-m model]")
        sys.exit(1)
    args = {}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in ("-d", "--data"):
            args["data_path"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] in ("-o", "--output"):
            args["output_path"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] in ("-b", "--backend"):
            args["backend"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] in ("-m", "--model"):
            args["model_name"] = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    run_extraction(**args)
