"""
Rule Definitions for RuleGround

Logical rules over predicates, organized by sport.
Rules from Paper Table 8 (Appendix F).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ruleground.predicates.ontology import Sport, SPORT_FROM_STR


@dataclass(frozen=True)
class Rule:
    """A logical rule over predicates."""

    name: str
    formula: str       # e.g., "contact_occurred AND NOT defender_set"
    sport: Sport
    outcome: str       # foul / violation type this rule identifies
    description: str = ""


# ---------------------------------------------------------------------------
# Rule Library -- Paper Table 8 (Appendix F) + additional implied rules
# ---------------------------------------------------------------------------

RULE_LIBRARY: List[Rule] = [
    # ---- Basketball ----
    Rule(
        name="blocking_foul",
        formula="contact_occurred AND NOT defender_set",
        sport=Sport.BASKETBALL,
        outcome="blocking",
        description="Defender not set at moment of contact -> blocking foul",
    ),
    Rule(
        name="charging_foul",
        formula="contact_occurred AND defender_set AND NOT restricted_area",
        sport=Sport.BASKETBALL,
        outcome="charging",
        description="Offensive player drives into set defender outside restricted area",
    ),
    Rule(
        name="travel",
        formula="pivot_foot_lifted AND NOT ball_released",
        sport=Sport.BASKETBALL,
        outcome="travel",
        description="Pivot foot lifted before ball is released",
    ),
    Rule(
        name="shooting_foul",
        formula="contact_occurred AND shooting_motion",
        sport=Sport.BASKETBALL,
        outcome="shooting_foul",
        description="Contact during shooting motion",
    ),
    Rule(
        name="legal_vertical_defense",
        formula="contact_occurred AND verticality_maintained AND defender_set",
        sport=Sport.BASKETBALL,
        outcome="no_foul",
        description="Defender stayed vertical in legal position; no foul",
    ),
    # ---- Football (American) ----
    Rule(
        name="dpi",
        formula="contact_before_arrival AND NOT incidental_contact AND ball_catchable",
        sport=Sport.FOOTBALL,
        outcome="DPI",
        description="Defensive pass interference: contact before ball arrives, "
        "non-incidental, on a catchable pass",
    ),
    Rule(
        name="opi",
        formula="contact_before_arrival AND offensive_push_off AND ball_in_air",
        sport=Sport.FOOTBALL,
        outcome="OPI",
        description="Offensive pass interference: push-off while ball is in the air",
    ),
    Rule(
        name="legal_contact_within_zone",
        formula="contact_occurred AND within_five_yards AND NOT contact_before_arrival",
        sport=Sport.FOOTBALL,
        outcome="no_foul",
        description="Legal contact within the 5-yard zone before ball is thrown",
    ),
    # ---- Soccer ----
    Rule(
        name="handball",
        formula="ball_contact_arm AND NOT arm_natural_position",
        sport=Sport.SOCCER,
        outcome="handball",
        description="Ball contacts arm in unnatural position",
    ),
    Rule(
        name="offside",
        formula="offside_position AND involved_in_play",
        sport=Sport.SOCCER,
        outcome="offside",
        description="Player in offside position actively involved in play",
    ),
    Rule(
        name="offside_exception_opponent",
        formula="offside_position AND played_by_opponent",
        sport=Sport.SOCCER,
        outcome="no_offside",
        description="Offside negated because ball was played by opponent",
    ),
    Rule(
        name="dogso",
        formula="denying_goal AND contact_occurred",
        sport=Sport.SOCCER,
        outcome="DOGSO",
        description="Denying an obvious goal-scoring opportunity via foul",
    ),
]

# ---------------------------------------------------------------------------
# Derived constants and helpers
# ---------------------------------------------------------------------------

RULE_REGISTRY: Dict[str, Rule] = {r.name: r for r in RULE_LIBRARY}


def get_rules_for_sport(sport: Sport | str) -> List[Rule]:
    """Return rules applicable to a given sport."""
    if isinstance(sport, str):
        sport = SPORT_FROM_STR[sport.lower()]
    return [r for r in RULE_LIBRARY if r.sport == sport]


def get_rule_names_for_sport(sport: Sport | str) -> List[str]:
    """Return rule names for a sport."""
    return [r.name for r in get_rules_for_sport(sport)]


def get_infraction_rules() -> List[Rule]:
    """Return rules whose outcome is an infraction (not no_foul/no_offside)."""
    return [r for r in RULE_LIBRARY if not r.outcome.startswith("no_")]


def get_all_referenced_predicates() -> List[str]:
    """Return all predicate names referenced by any rule formula."""
    preds = set()
    for rule in RULE_LIBRARY:
        tokens = rule.formula.replace("(", " ").replace(")", " ").split()
        for token in tokens:
            if token not in ("AND", "OR", "NOT"):
                preds.add(token)
    return sorted(preds)
