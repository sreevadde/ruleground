"""
SPORTU → RuleGround Data Converter

Converts the SPORTU benchmark (chili-lab/SPORTU, ICLR 2025) into the
RuleGround annotation format for training and evaluation.

SPORTU provides:
    - MC questions: foul Y/N, foul type classification, difficulty levels
    - OE questions: free-text rationales explaining violations
    - Videos: slow-motion clips already cropped to the event

This converter produces:
    - train.json, val.json, test.json with RuleGround schema
    - Unified foul taxonomy across basketball, soccer, football
    - Rationale text for predicate extraction pipeline

Usage:
    python -m ruleground.data.preparation.sportu_converter \\
        --sportu-dir data/sportu_raw \\
        --output-dir data/sportr \\
        --video-dir data/sportu_raw/videos
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SPORTU sport prefix → RuleGround sport name
# ---------------------------------------------------------------------------

SPORTU_PREFIX_MAP = {
    "Basketball": "basketball",
    "soccer": "soccer",
    "american_football": "football",
}

# ---------------------------------------------------------------------------
# Unified foul taxonomy: SPORTU foul name → (q2_class_index, canonical_name)
#
# Class 0 = no infraction (clean play)
# Classes 1-16 = foul types across all 3 sports
# ---------------------------------------------------------------------------

FOUL_TAXONOMY: Dict[str, Tuple[int, str]] = {
    # Basketball fouls (classes 1-7)
    "blocking foul":                (1, "blocking_foul"),
    "flagrant foul":                (2, "flagrant_foul"),
    "illegal contact foul":         (3, "illegal_contact"),
    "hand-check foul":              (3, "illegal_contact"),  # merge with illegal contact
    "holding foul":                 (4, "holding"),
    "illegal elbow foul":           (5, "illegal_elbow"),
    "neck contact foul":            (5, "illegal_elbow"),    # merge: upper body contact
    "head contact foul":            (5, "illegal_elbow"),    # merge: upper body contact
    "illegal shoulder charge foul": (6, "charging"),
    "traveling violation":          (7, "traveling"),

    # Soccer fouls (classes 8-14)
    "handball":                     (8, "handball"),
    "handball foul":                (8, "handball"),
    "tackle foul":                  (9, "tackle"),
    "kick foul":                    (9, "tackle"),           # merge: leg fouls
    "step foul":                    (9, "tackle"),           # merge: leg fouls
    "trip foul":                    (9, "tackle"),           # merge: leg fouls
    "simulation":                   (10, "simulation"),
    "dive foul":                    (10, "simulation"),      # merge: simulation
    "offside":                      (11, "offside"),
    "offside foul":                 (11, "offside"),
    "dogso-spa":                    (12, "dogso"),
    "pull foul":                    (13, "pull_push"),
    "push foul":                    (13, "pull_push"),
    "collide foul":                 (13, "pull_push"),       # merge: body contact
    "elbow foul":                   (13, "pull_push"),       # merge: body contact
    "block foul":                   (13, "pull_push"),       # merge: body contact
    "move foul":                    (13, "pull_push"),       # merge: body contact
    "holding":                      (14, "holding_soccer"),

    # Football fouls (classes 15-16)
    "pass interference":            (15, "pass_interference"),
    "personal foul":                (16, "personal_foul"),
    "false start":                  (16, "personal_foul"),   # merge: procedural fouls
    "offsides":                     (16, "personal_foul"),   # merge: procedural fouls
    "holding_football":             (16, "personal_foul"),   # merge: procedural fouls
}

# Names that indicate NOT a foul (q1=0, q2=0)
NON_FOUL_ANSWERS = {
    "none of the above", "missed tackle", "interception",
    "fumble", "missed catch", "touchdown", "penalty kick",
}

Q2_CLASS_NAMES = {
    0: "no_infraction",
    1: "blocking_foul",
    2: "flagrant_foul",
    3: "illegal_contact",
    4: "holding",
    5: "illegal_elbow",
    6: "charging",
    7: "traveling",
    8: "handball",
    9: "tackle",
    10: "simulation",
    11: "offside",
    12: "dogso",
    13: "pull_push",
    14: "holding_soccer",
    15: "pass_interference",
    16: "personal_foul",
}


def classify_foul(answer_text: str, sport_prefix: str) -> Tuple[int, str]:
    """Map a SPORTU foul answer to (q2_class_index, canonical_name).

    Args:
        answer_text:   The correct answer option text from SPORTU MC.
        sport_prefix:  SPORTU sport prefix (e.g., 'Basketball').

    Returns:
        (q2_label, canonical_name). Returns (0, "no_infraction") for non-fouls.
    """
    normalized = answer_text.strip().lower()

    if normalized in NON_FOUL_ANSWERS:
        return 0, "no_infraction"

    # Special case: "Holding" in football context
    if normalized == "holding" and sport_prefix == "american_football":
        return 16, "personal_foul"

    if normalized in FOUL_TAXONOMY:
        return FOUL_TAXONOMY[normalized]

    # Fuzzy match: check if any known foul is a substring
    for known, (idx, name) in FOUL_TAXONOMY.items():
        if known in normalized or normalized in known:
            return idx, name

    logger.warning(f"Unknown foul type: '{answer_text}' (sport={sport_prefix})")
    return 0, "no_infraction"


# ---------------------------------------------------------------------------
# Question classification helpers
# ---------------------------------------------------------------------------

def is_foul_yn_question(question: str) -> bool:
    """Check if a MC question asks about foul presence (yes/no)."""
    q = question.lower()
    return any(x in q for x in [
        "is there a foul", "did a foul", "was there a foul",
        "is this a foul", "foul occur", "infraction",
        "rule violation occur", "is there a rule violation",
        "is there a rule infraction",
    ])


def is_foul_type_question(question: str) -> bool:
    """Check if a MC question asks about foul type classification."""
    q = question.lower()
    return any(x in q for x in [
        "what specific type", "what kind of", "what type of foul",
        "which of the following descriptions", "what specific foul",
    ])


def parse_foul_yn(entry: Dict) -> Optional[int]:
    """Parse a foul Y/N MC entry to q1_label.

    Returns:
        1 for infraction, 0 for no infraction, None if unparseable.
    """
    correct = entry["options"][entry["answer"]].strip().lower()
    if correct in ("yes", "yes, there is a foul", "yes, there is a rule violation"):
        return 1
    if correct in ("no", "no foul", "no, there is no foul", "no, there is no rule violation"):
        return 0
    if "yes" in correct:
        return 1
    if "no" in correct:
        return 0
    return None


# ---------------------------------------------------------------------------
# Video ID parsing
# ---------------------------------------------------------------------------

def parse_video_id(entry_id: str) -> str:
    """Extract base video ID from SPORTU entry ID.

    SPORTU IDs: '{sport}_{number}' or '{sport}_{number}_{question_variant}'
    We want the base: '{sport}_{number}'
    """
    parts = entry_id.split("_")

    # Handle multi-word sport prefixes: 'american_football_123_0'
    if entry_id.startswith("american_football"):
        # 'american_football_123' or 'american_football_123_0'
        rest = entry_id[len("american_football_"):]
        rest_parts = rest.split("_")
        if len(rest_parts) >= 2 and rest_parts[-1].isdigit() and rest_parts[0].isdigit():
            # Has question variant suffix
            return f"american_football_{rest_parts[0]}"
        return f"american_football_{rest_parts[0]}"

    # Single-word prefix: 'Basketball_85_1' or 'soccer_260_2'
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-1])

    return entry_id


def get_sport_prefix(entry_id: str) -> Optional[str]:
    """Get SPORTU sport prefix from entry ID."""
    for prefix in SPORTU_PREFIX_MAP:
        if entry_id.startswith(prefix):
            return prefix
    return None


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

class SportUConverter:
    """Convert SPORTU benchmark data to RuleGround annotation format.

    Args:
        sportu_dir:  Path to directory containing SPORTU JSON files.
        video_dir:   Path to directory containing SPORTU video files.
                     Videos should be organized as {sport}/{video_id}.mp4
                     or flat as {video_id}.mp4.
        output_dir:  Path to output RuleGround data directory.
    """

    def __init__(
        self,
        sportu_dir: str | Path,
        video_dir: Optional[str | Path] = None,
        output_dir: str | Path = "data/sportr",
    ):
        self.sportu_dir = Path(sportu_dir)
        self.video_dir = Path(video_dir) if video_dir else None
        self.output_dir = Path(output_dir)

        self.mc_data: List[Dict] = []
        self.oe_data: List[Dict] = []

    def load(self) -> None:
        """Load SPORTU annotation files."""
        mc_path = self.sportu_dir / "SportU_Video_mc.json"
        oe_path = self.sportu_dir / "SportU_video_oe.json"

        with open(mc_path) as f:
            self.mc_data = json.load(f)
        logger.info(f"Loaded {len(self.mc_data)} MC entries from {mc_path}")

        with open(oe_path) as f:
            self.oe_data = json.load(f)
        logger.info(f"Loaded {len(self.oe_data)} OE entries from {oe_path}")

    def _build_video_records(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate per-video information from all MC and OE entries.

        Returns:
            Dict mapping video_id → {sport, q1_label, q2_label, rationale, ...}
        """
        records: Dict[str, Dict[str, Any]] = {}

        # Process MC entries
        for entry in self.mc_data:
            sport_prefix = get_sport_prefix(entry["id"])
            if sport_prefix is None or sport_prefix not in SPORTU_PREFIX_MAP:
                continue

            video_id = parse_video_id(entry["id"])
            sport = SPORTU_PREFIX_MAP[sport_prefix]

            if video_id not in records:
                records[video_id] = {
                    "video_id": video_id,
                    "sport": sport,
                    "sport_prefix": sport_prefix,
                    "q1_label": None,
                    "q2_label": None,
                    "q2_name": None,
                    "rationale": None,
                    "difficulty": entry.get("difficulty level", "medium"),
                    "mc_count": 0,
                }

            records[video_id]["mc_count"] += 1

            # Extract Q1 (foul yes/no)
            if is_foul_yn_question(entry["question"]):
                q1 = parse_foul_yn(entry)
                if q1 is not None:
                    records[video_id]["q1_label"] = q1

            # Extract Q2 (foul type)
            if is_foul_type_question(entry["question"]):
                correct_text = entry["options"][entry["answer"]]
                q2_idx, q2_name = classify_foul(correct_text, sport_prefix)
                if records[video_id]["q2_label"] is None or q2_idx > 0:
                    # Prefer non-zero classification over "none"
                    records[video_id]["q2_label"] = q2_idx
                    records[video_id]["q2_name"] = q2_name

        # Process OE entries for rationales
        for entry in self.oe_data:
            sport_prefix = get_sport_prefix(entry["id"])
            if sport_prefix is None or sport_prefix not in SPORTU_PREFIX_MAP:
                continue

            video_id = parse_video_id(entry["id"])
            if video_id in records and entry.get("answer"):
                # Prefer longer rationales (more detail)
                existing = records[video_id].get("rationale") or ""
                if len(entry["answer"]) > len(existing):
                    records[video_id]["rationale"] = entry["answer"]

        return records

    def _resolve_labels(self, records: Dict[str, Dict]) -> List[Dict]:
        """Resolve missing labels and produce final annotation list.

        Inference rules:
        - If q1_label is None but q2_label > 0 → q1_label = 1 (has foul type → infraction)
        - If q1_label is None and q2_label == 0 → q1_label = 0
        - If q2_label is None and q1_label == 0 → q2_label = 0
        - If q2_label is None and q1_label == 1 → skip (ambiguous)

        Returns:
            List of resolved annotation dicts.
        """
        annotations = []
        stats = {"total": 0, "resolved": 0, "skipped": 0, "infraction": 0, "clean": 0}

        for video_id, rec in records.items():
            stats["total"] += 1
            q1 = rec["q1_label"]
            q2 = rec["q2_label"]

            # Resolve q1 from q2
            if q1 is None and q2 is not None:
                q1 = 1 if q2 > 0 else 0
            # Resolve q2 from q1
            if q2 is None and q1 == 0:
                q2 = 0
            # q1=0 overrides q2: if Y/N says no foul, foul type is irrelevant
            # (e.g., "handball but arm in natural position" = not an infraction)
            if q1 == 0:
                q2 = 0
            # Skip ambiguous
            if q1 is None or q2 is None:
                stats["skipped"] += 1
                continue

            # Q5: entire clip is the event (SPORTU clips are pre-cropped)
            q5_span = [0.0, 1.0] if q1 > 0 else None

            annotations.append({
                "video_id": video_id,
                "sport": rec["sport"],
                "q1_label": q1,
                "q2_label": q2,
                "q5_span": q5_span,
                "rationale": rec.get("rationale"),
            })

            stats["resolved"] += 1
            if q1 > 0:
                stats["infraction"] += 1
            else:
                stats["clean"] += 1

        logger.info(
            f"Resolved {stats['resolved']}/{stats['total']} videos "
            f"({stats['infraction']} infractions, {stats['clean']} clean, "
            f"{stats['skipped']} skipped)"
        )
        return annotations

    def _check_videos(self, annotations: List[Dict]) -> List[Dict]:
        """Filter annotations to only those with available video files.

        Checks for video files in video_dir with patterns:
            {video_id}.mp4
            {sport}/{video_id}.mp4
            {sport}/{number}.mp4
        """
        if self.video_dir is None:
            logger.info("No video_dir specified, keeping all annotations")
            return annotations

        found = []
        missing = 0

        for anno in annotations:
            vid = anno["video_id"]
            candidates = [
                self.video_dir / f"{vid}.mp4",
                self.video_dir / anno["sport"] / f"{vid}.mp4",
            ]
            # Try numeric-only filename: 'Basketball_85' → 'basketball/85.mp4'
            parts = vid.split("_")
            if len(parts) >= 2:
                num = parts[-1]
                candidates.append(self.video_dir / anno["sport"] / f"{num}.mp4")

            if any(p.exists() for p in candidates):
                # Store the path that works
                for p in candidates:
                    if p.exists():
                        anno["video_path"] = str(p)
                        break
                found.append(anno)
            else:
                missing += 1

        logger.info(f"Videos found: {len(found)}, missing: {missing}")
        return found

    def _split_data(
        self,
        annotations: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split annotations into train/val/test by video.

        Stratified by sport and q1_label to maintain balance.

        Returns:
            (train, val, test) annotation lists.
        """
        rng = random.Random(seed)

        # Group by (sport, q1_label) for stratification
        groups: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
        for anno in annotations:
            key = (anno["sport"], anno["q1_label"])
            groups[key].append(anno)

        train, val, test = [], [], []

        for key, items in groups.items():
            rng.shuffle(items)
            n = len(items)
            n_val = max(1, int(n * val_ratio))
            n_test = max(1, int(n * (1 - train_ratio - val_ratio)))
            n_train = n - n_val - n_test

            train.extend(items[:n_train])
            val.extend(items[n_train:n_train + n_val])
            test.extend(items[n_train + n_val:])

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def convert(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        require_videos: bool = False,
    ) -> Dict[str, Any]:
        """Run the full conversion pipeline.

        Args:
            train_ratio:    Fraction for training set.
            val_ratio:      Fraction for validation set.
            seed:           Random seed for splits.
            require_videos: If True, filter to only samples with video files.

        Returns:
            Summary statistics dict.
        """
        self.load()
        records = self._build_video_records()
        annotations = self._resolve_labels(records)

        if require_videos:
            annotations = self._check_videos(annotations)

        train, val, test = self._split_data(annotations, train_ratio, val_ratio, seed)

        # Write output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        anno_dir = self.output_dir / "annotations"
        anno_dir.mkdir(parents=True, exist_ok=True)

        for split_name, data in [("train", train), ("val", val), ("test", test)]:
            # Strip internal fields before saving
            clean = []
            for d in data:
                clean.append({
                    "video_id": d["video_id"],
                    "sport": d["sport"],
                    "q1_label": d["q1_label"],
                    "q2_label": d["q2_label"],
                    "q5_span": d["q5_span"],
                    "rationale": d.get("rationale"),
                })
            path = anno_dir / f"{split_name}.json"
            with open(path, "w") as f:
                json.dump(clean, f, indent=2)
            logger.info(f"Wrote {len(clean)} annotations to {path}")

        # Write taxonomy reference
        taxonomy_path = self.output_dir / "foul_taxonomy.json"
        with open(taxonomy_path, "w") as f:
            json.dump({str(k): v for k, v in Q2_CLASS_NAMES.items()}, f, indent=2)

        # Compute stats
        from collections import Counter
        all_data = train + val + test
        sport_counts = Counter(d["sport"] for d in all_data)
        q1_counts = Counter(d["q1_label"] for d in all_data)
        q2_counts = Counter(d["q2_label"] for d in all_data)
        rationale_count = sum(1 for d in all_data if d.get("rationale"))

        stats = {
            "total_samples": len(all_data),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "by_sport": dict(sport_counts),
            "q1_distribution": {0: q1_counts[0], 1: q1_counts[1]},
            "q2_classes_used": len(set(d["q2_label"] for d in all_data)),
            "q2_distribution": dict(q2_counts.most_common()),
            "rationale_count": rationale_count,
            "rationale_coverage": rationale_count / len(all_data) if all_data else 0,
        }

        # Write stats
        stats_path = self.output_dir / "conversion_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        return stats

    def create_video_symlinks(self) -> int:
        """Create symlinks in output_dir/videos/ pointing to source videos.

        Returns:
            Number of symlinks created.
        """
        if self.video_dir is None:
            logger.warning("No video_dir specified")
            return 0

        videos_out = self.output_dir / "videos"
        videos_out.mkdir(parents=True, exist_ok=True)

        created = 0
        for anno_path in (self.output_dir / "annotations").glob("*.json"):
            with open(anno_path) as f:
                annotations = json.load(f)
            for anno in annotations:
                vid = anno["video_id"]
                target = videos_out / f"{vid}.mp4"
                if target.exists():
                    continue

                # Search for source video
                candidates = [
                    self.video_dir / f"{vid}.mp4",
                    self.video_dir / anno["sport"] / f"{vid}.mp4",
                ]
                parts = vid.split("_")
                if len(parts) >= 2:
                    candidates.append(
                        self.video_dir / anno["sport"] / f"{parts[-1]}.mp4"
                    )

                for src in candidates:
                    if src.exists():
                        target.symlink_to(src.resolve())
                        created += 1
                        break

        logger.info(f"Created {created} video symlinks in {videos_out}")
        return created


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert SPORTU benchmark to RuleGround format"
    )
    parser.add_argument(
        "--sportu-dir", type=str, default="data/sportu_raw",
        help="Directory containing SPORTU JSON files",
    )
    parser.add_argument(
        "--video-dir", type=str, default=None,
        help="Directory containing SPORTU video files (optional)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/sportr",
        help="Output directory for RuleGround annotations",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7,
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--require-videos", action="store_true",
        help="Only include samples where video files exist",
    )
    parser.add_argument(
        "--symlink-videos", action="store_true",
        help="Create symlinks to videos in output directory",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

    converter = SportUConverter(
        sportu_dir=args.sportu_dir,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
    )
    stats = converter.convert(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        require_videos=args.require_videos,
    )

    if args.symlink_videos and args.video_dir:
        converter.create_video_symlinks()

    print("\n=== Conversion Summary ===")
    print(f"Total samples:  {stats['total_samples']}")
    print(f"  Train:        {stats['train']}")
    print(f"  Val:          {stats['val']}")
    print(f"  Test:         {stats['test']}")
    print(f"\nBy sport:")
    for sport, cnt in sorted(stats["by_sport"].items()):
        print(f"  {sport}: {cnt}")
    print(f"\nQ1 (infraction Y/N): {stats['q1_distribution']}")
    print(f"Q2 classes used: {stats['q2_classes_used']}")
    print(f"Rationale coverage: {stats['rationale_coverage']:.1%}")


if __name__ == "__main__":
    main()
